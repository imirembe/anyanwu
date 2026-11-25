import Foundation
import CryptoKit
import AVFoundation
import CoreML

struct AppleCheckFailure: Error { let message: String }
func require(_ value: Bool, _ message: String) throws {
    if !value { throw AppleCheckFailure(message: message) }
}

@main struct AppleChecks {
    static func main() async {
        do { try await run() }
        catch { print("FAIL: \(error)"); exit(1) }
    }
    static func run() async throws {
        let arguments = CommandLine.arguments
        let root = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let vocabularyURL = URL(fileURLWithPath: arguments[2])
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let lexicon = try SentencePieceLexicon(data: Data(contentsOf: vocabularyURL))
        try require(lexicon.pieces.count == 10_288, "real SentencePiece vocabulary count")
        let dotted = ["ị", "ọ", "ụ", "ṅ"]
        let ids = try dotted.map { letter -> Int in
            guard let id = lexicon.pieces.firstIndex(where: { $0.text == letter }) else {
                throw AppleCheckFailure(message: "Igbo letter missing from tokenizer: \(letter)")
            }
            return id
        }
        try require(try lexicon.decode(ids) == dotted.joined(), "Igbo diacritics changed during decoding")
        try require(AppleCTC.collapse([4, 4, 1, 4, 5, 5, 1]) == [4, 1, 4, 5, 1], "CTC blank-separated repeat lost")
        try require(try lexicon.decode([0, 1, 2, 3] + ids) == dotted.joined(), "special token removal")
        for invalid in [Data(), Data([10, 127]), Data(repeating: 255, count: 10)] {
            do { _ = try SentencePieceLexicon(data: invalid); throw AppleCheckFailure(message: "invalid protobuf accepted") }
            catch is AppleSpeechError { }
        }
        let normalized = try AppleCTC.normalizedWindow([Float(1), 2, 3, 4][...])
        try require(normalized.count == 160_000 && abs(normalized.prefix(4).reduce(0, +)) < 0.0001,
                    "normalization/padding order")
        try require(normalized.dropFirst(4).allSatisfy { $0 == 0 }, "padding changed")
        print("PASS: actual tokenizer preserves Igbo underdots; bounded protobuf, CTC repeats and waveform normalization")

        for type in [MLMultiArrayDataType.float32, .float16] {
            let logits = try MLMultiArray(shape: [1, 4, 10288], dataType: type)
            for index in 0..<logits.count { logits[index] = -1 }
            for (frame, token) in [4, 4, 1, 5].enumerated() { logits[[0, NSNumber(value: frame), NSNumber(value: token)]] = 2 }
            try require(try AppleSpeechRecognizer.tokenIDs(logits, validSamples: 1360) == [4, 4, 1, 5], "Core ML argmax differs by dtype")
            try require(try AppleSpeechRecognizer.tokenIDs(logits, validSamples: 720) == [4, 4], "padded output frames were decoded")
        }
        let waveform = root.appending(path: "test.wav")
        let audioFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
        let pcm = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: 24000)!
        pcm.frameLength = 24000
        for frame in 0..<24000 { pcm.floatChannelData![0][frame] = Float(sin(Double(frame) * 2 * .pi * 440 / 24000) * 0.1) }
        try autoreleasepool {
            let audio = try AVAudioFile(forWriting: waveform, settings: audioFormat.settings)
            try audio.write(from: pcm)
        }
        let resampled = try AppleSpeechRecognizer.readAudio(waveform)
        try require((15900...16100).contains(resampled.count), "AVFoundation failed 24kHz to 16kHz conversion: \(resampled.count) samples")
        print("PASS: real Core ML Float16/Float32 output decoding crops padding; AVFoundation resamples bounded audio")

        let spec = AppleSpeechAssets.files.first { $0.path == "tokenizer.model" }!
        try AppleSpeechAssets.verify(vocabularyURL, spec: spec)
        let corrupt = root.appending(path: "corrupt.model")
        var data = try Data(contentsOf: vocabularyURL); data[0] ^= 1
        try data.write(to: corrupt)
        do { try AppleSpeechAssets.verify(corrupt, spec: spec); throw AppleCheckFailure(message: "corrupt digest accepted") }
        catch AppleSpeechError.invalidDownload { }
        let assets = AppleSpeechAssets(directory: root)
        try require(try await assets.installed() == false, "empty install marked ready")
        try require(AppleSpeechAssets.downloadsAllowed == false, "Mac can download full weights")
        do {
            try await assets.install { _ in }
            throw AppleCheckFailure(message: "model downloaded on Mac")
        } catch AppleSpeechError.physicalDevice { }
        try require(AppleSpeechAssets.downloadBytes < ResourcePolicy.onDeviceDownloadBytes, "download exceeds its budget")
        let savedText = root.appending(path: "history-sentinel.txt")
        try Data("preserve history".utf8).write(to: savedText)
        let installedDirectory = root.appending(path: "OnDeviceSpeech/active", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: installedDirectory, withIntermediateDirectories: true)
        try Data([1]).write(to: installedDirectory.appending(path: "model-fixture"))
        try await assets.remove()
        try require(!FileManager.default.fileExists(atPath: installedDirectory.path), "explicit model removal left files behind")
        try require(try String(contentsOf: savedText, encoding: .utf8) == "preserve history", "model removal deleted unrelated history")
        let orphan = root.appending(path: "OnDeviceSpeech/staging-abandoned", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)
        _ = try await AppleSpeechAssets(directory: root).installed()
        try require(!FileManager.default.fileExists(atPath: orphan.path), "abandoned install staging survived startup")
        print("PASS: pinned hash verification rejects corruption; model download is refused on this Mac")

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ModelFixtureProtocol.self]
        for scenario in ["small", "declared", "streamed", "truncated"] {
            let target = root.appending(path: scenario)
            do {
                try await BoundedHTTP.download(URL(string: "https://fixture.invalid/\(scenario)")!, to: target,
                                                 bytes: 5, configuration: config) { _ in }
                try require(scenario == "small", "invalid stream accepted")
                try require(try Data(contentsOf: target) == Data("Ndewo".utf8), "streamed bytes changed")
            } catch AppleSpeechError.invalidDownload {
                try require(scenario != "small", "valid stream rejected")
                try require(!FileManager.default.fileExists(atPath: target.path), "partial model left behind")
            }
        }
        let partial = root.appending(path: "cancelled")
        let download = Task {
            try await BoundedHTTP.download(URL(string: "https://fixture.invalid/pending")!, to: partial,
                                             bytes: 5, configuration: config) { _ in }
        }
        try await Task.sleep(for: .milliseconds(40)); download.cancel()
        do { try await download.value; throw AppleCheckFailure(message: "download ignored cancellation") }
        catch is CancellationError { }
        try require(!FileManager.default.fileExists(atPath: partial.path), "cancelled model left behind")
        print("PASS: model downloads stream to disk, enforce declared/streamed sizes, and erase failed/cancelled staging files")

        try require(AssistantSettings().mode == .apple, "new installation does not prefer Apple")
        try require(AssistantSettings().automaticallyPrepareSpeech, "first launch does not prepare speech")
        let legacy = try JSONDecoder().decode(AssistantSettings.self, from: Data(#"{"mode":"Preview","saveHistory":false}"#.utf8))
        try require(legacy.mode == .apple && !legacy.saveHistory, "Apple migration changed privacy preferences")
        var preview = AssistantSettings(); preview.mode = .preview
        let restored = try JSONDecoder().decode(AssistantSettings.self, from: JSONEncoder().encode(preview))
        try require(restored.mode == .preview, "explicit new preview selection lost")
        let connected = try JSONDecoder().decode(AssistantSettings.self, from: Data(#"{"mode":"Connected","serverURL":"https://voice.example"}"#.utf8))
        try require(connected.mode == .connected && connected.serverURL == "https://voice.example", "service connection lost")
        print("PASS: Apple is the default; previous connection/privacy preferences and explicit preview selections survive")

        var optedOut = AssistantSettings(); optedOut.automaticallyPrepareSpeech = false
        let savedOptOut = try JSONDecoder().decode(AssistantSettings.self, from: JSONEncoder().encode(optedOut))
        try require(!savedOptOut.automaticallyPrepareSpeech, "explicit download cancellation/removal opt-out was lost")
        var policy = SpeechPreparationPolicy()
        policy.activate(enabled: true)
        try require(!policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "download started before Wi-Fi")
        policy.updateNetwork(wifi: true)
        try require(!policy.claimAttempt(installed: true, installing: false, downloadsAllowed: true), "installed model was downloaded again")
        try require(!policy.claimAttempt(installed: false, installing: true, downloadsAllowed: true), "duplicate download started")
        try require(!policy.claimAttempt(installed: false, installing: false, downloadsAllowed: false), "automatic setup bypassed device guard")
        try require(policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "first Wi-Fi setup did not start")
        policy.activate(enabled: true)
        try require(!policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "same foreground session retried in a loop")
        policy.suspend()
        policy.updateNetwork(wifi: true)
        try require(!policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "background update started a download")
        policy.activate(enabled: true)
        try require(policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "returning foreground could not retry")
        policy.updateNetwork(wifi: false); policy.updateNetwork(wifi: true)
        try require(policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "Wi-Fi reconnection could not retry")
        policy.activate(enabled: savedOptOut.automaticallyPrepareSpeech)
        policy.suspend(); policy.activate(enabled: savedOptOut.automaticallyPrepareSpeech); policy.updateNetwork(wifi: true)
        try require(!policy.claimAttempt(installed: false, installing: false, downloadsAllowed: true), "cancelled/removed model downloaded on foreground return")
        print("PASS: automatic preparation waits for Wi-Fi, avoids duplicate/repeated downloads, retries after interruption and preserves explicit opt-out")
    }
}

final class ModelFixtureProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let scenario = request.url!.lastPathComponent
        if scenario == "pending" { return }
        let headers = scenario == "declared" ? ["Content-Length": "1000000"] : [:]
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: headers)!, cacheStoragePolicy: .notAllowed)
        let bytes = scenario == "streamed" ? Data(repeating: 1, count: 6)
            : scenario == "truncated" ? Data([1]) : Data("Ndewo".utf8)
        client?.urlProtocol(self, didLoad: bytes)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
