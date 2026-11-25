import Foundation

@MainActor final class AppleAssistant {
    var capabilities = AppleCapabilities()
    var isSpeaking = false
    var speakingMessageID: UUID?
    var isInstalling = false
    init(directory: URL) {}
    func activatePreparation(automatically: Bool, language: InterfaceLanguage) {}
    func installSpeech(language: InterfaceLanguage) {}
    func cancelInstall() {}
    func requireRecognition() async throws { throw AppleSpeechError.notInstalled }
    func transcribe(_ url: URL) async throws -> String { throw AppleSpeechError.notInstalled }
    func reply(to text: String, context: [TurnRequest.ContextMessage], language: InterfaceLanguage) async throws -> AppleReply {
        AppleReply(text: PreviewReplies.reply(to: text), isPreview: true, notice: nil)
    }
    func speak(_ text: String, messageID: UUID) throws { throw AppleSpeechError.speechUnavailable }
    func stopSpeaking() { isSpeaking = false; speakingMessageID = nil }
    func releaseResources() { stopSpeaking() }
    func removeSpeech(language: InterfaceLanguage) async {}
}

// macOS checks exercise the actual store; AVFoundation recording/playback stays
// covered by the iOS targets and needs device testing. No microphone is opened.
@MainActor final class VoiceRecorder {
    var isRecording = false
    var elapsed: TimeInterval = 1
    var onLimitReached: (() -> Void)?
    var onInterruption: (() -> Void)?
    private var url: URL?
    func start(at url: URL) async throws {
        try Task.checkCancellation()
        try Data([1, 2]).write(to: url)
        self.url = url; isRecording = true
    }
    func stop() -> URL? { defer { url = nil; isRecording = false }; return url }
    func cancel() { if let url = stop() { try? FileManager.default.removeItem(at: url) } }
    enum VoiceError: Error {
        case failed
        func message(in language: InterfaceLanguage) -> String { "fixture" }
    }
}

@MainActor final class VoicePlayback {
    var playingFilename: String?
    var onReleased: ((String) -> Void)?
    func toggle(url: URL) throws {
        if playingFilename == url.lastPathComponent { stop(); return }
        stop(); playingFilename = url.lastPathComponent
    }
    func stop() {
        let filename = playingFilename; playingFilename = nil
        if let filename { onReleased?(filename) }
    }
}
