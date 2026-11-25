import AVFoundation
import Observation

@MainActor @Observable
final class VoiceRecorder: NSObject, AVAudioRecorderDelegate {
    var isRecording = false
    var elapsed: TimeInterval = 0
    var level: Double = 0
    var wasInterrupted = false
    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    private var currentURL: URL?
    private var interruptionObserver: NSObjectProtocol?
    var onLimitReached: (() -> Void)?
    var onInterruption: (() -> Void)?

    override init() {
        super.init()
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isRecording else { return }
                self.wasInterrupted = true
                self.onInterruption?()
            }
        }
    }

    isolated deinit {
        timer?.invalidate()
        if let interruptionObserver { NotificationCenter.default.removeObserver(interruptionObserver) }
    }

    func start(at url: URL) async throws {
        try Task.checkCancellation()
        let permitted = await AVAudioApplication.requestRecordPermission()
        try Task.checkCancellation()
        guard permitted else { throw VoiceError.permission }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC), AVSampleRateKey: 24_000,
            AVNumberOfChannelsKey: 1, AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        let recording = try AVAudioRecorder(url: url, settings: settings)
        recording.delegate = self
        recording.isMeteringEnabled = true
        guard recording.record() else { throw VoiceError.failed }
        recorder = recording
        currentURL = url
        elapsed = 0
        wasInterrupted = false
        isRecording = true
        timer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let recorder = self.recorder, self.isRecording else { return }
                recorder.updateMeters()
                self.elapsed = recorder.currentTime
                self.level = max(0.04, min(1, pow(10, Double(recorder.averagePower(forChannel: 0)) / 35)))
                if self.elapsed >= 60 { self.onLimitReached?() }
            }
        }
    }

    func stop() -> URL? {
        timer?.invalidate()
        timer = nil
        recorder?.stop()
        recorder = nil
        isRecording = false
        level = 0
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        let url = currentURL
        currentURL = nil
        if let url { try? FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: url.path) }
        return url
    }

    func cancel() {
        if let url = stop() { try? FileManager.default.removeItem(at: url) }
    }

    enum VoiceError: LocalizedError {
        case permission, failed
        var errorDescription: String? {
            message(in: .english)
        }
        func message(in language: InterfaceLanguage) -> String {
            switch self {
            case .permission: language.text("Microphone access is off. Enable it for Anyanwụ in iPhone Settings to record your voice.", "Enyebeghị Anyanwụ ohere iji igwe okwu. Gbanwee ya na Ntọala iPhone ka ịdekọọ olu gị.")
            case .failed: language.text("Couldn’t start recording. Check your microphone and try again.", "Enweghị ike ịmalite ndekọ. Lelee igwe okwu gị, wee nwaa ọzọ.")
            }
        }
    }
}

@MainActor @Observable
final class VoicePlayback: NSObject, AVAudioPlayerDelegate {
    var playingFilename: String?
    private var player: AVAudioPlayer?
    var onReleased: ((String) -> Void)?

    func toggle(url: URL) throws {
        if playingFilename == url.lastPathComponent { stop(); return }
        stop()
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio)
        try session.setActive(true)
        let player = try AVAudioPlayer(contentsOf: url)
        player.delegate = self
        guard player.play() else { throw VoiceRecorder.VoiceError.failed }
        self.player = player
        playingFilename = url.lastPathComponent
    }

    func stop() {
        let released = playingFilename
        player?.stop()
        player?.delegate = nil
        player = nil
        playingFilename = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        if let released { onReleased?(released) }
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard let self, self.player === player else { return }
            self.stop()
        }
    }
}
