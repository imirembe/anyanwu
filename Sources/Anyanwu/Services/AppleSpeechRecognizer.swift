import AVFoundation
import CoreML
import Foundation

/// One model instance, one inference at a time. Actor isolation keeps file I/O,
/// waveform conversion and Core ML predictions away from the UI thread.
actor AppleSpeechRecognizer {
    private let assets: AppleSpeechAssets
    private var model: MLModel?
    private var lexicon: SentencePieceLexicon?

    init(assets: AppleSpeechAssets) { self.assets = assets }
    func unload() { model = nil; lexicon = nil }

    func prepare() async throws {
        try Task.checkCancellation()
        if model == nil {
            let locations = try await assets.locations()
            try Task.checkCancellation()
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let loaded = try MLModel(contentsOf: locations.model, configuration: configuration)
            guard loaded.modelDescription.inputDescriptionsByName["audio"]?.multiArrayConstraint?.shape.map(\.intValue) == [1, ResourcePolicy.onDeviceInputSamples],
                  loaded.modelDescription.outputDescriptionsByName["logits"] != nil else { throw AppleSpeechError.invalidModel }
            let vocabulary = try SentencePieceLexicon(data: Data(contentsOf: locations.vocabulary))
            try Task.checkCancellation()
            model = loaded; lexicon = vocabulary
        }
    }

    func transcribe(_ url: URL) async throws -> String {
        try await prepare()
        guard let model, let lexicon else { throw AppleSpeechError.invalidModel }
        let samples = try Self.readAudio(url)
        var parts: [String] = []
        for start in stride(from: 0, to: samples.count, by: ResourcePolicy.onDeviceInputSamples) {
            try Task.checkCancellation()
            let end = min(samples.count, start + ResourcePolicy.onDeviceInputSamples)
            let text: String = try autoreleasepool {
                let window = try AppleCTC.normalizedWindow(samples[start..<end])
                let input = try MLMultiArray(shape: [1, NSNumber(value: window.count)], dataType: .float32)
                window.withUnsafeBufferPointer { source in
                    input.dataPointer.assumingMemoryBound(to: Float.self).update(from: source.baseAddress!, count: window.count)
                }
                let features = try MLDictionaryFeatureProvider(dictionary: ["audio": input])
                let result = try model.prediction(from: features)
                try Task.checkCancellation()
                guard let logits = result.featureValue(for: "logits")?.multiArrayValue else { throw AppleSpeechError.invalidModel }
                let ids = try Self.tokenIDs(logits, validSamples: end - start)
                return try lexicon.decode(AppleCTC.collapse(ids))
            }
            if !text.isEmpty { parts.append(text) }
            guard parts.reduce(0, { $0 + $1.count }) <= 8_000 else { throw AppleSpeechError.invalidModel }
        }
        try Task.checkCancellation()
        let text = parts.joined(separator: " ").precomposedStringWithCanonicalMapping
        guard !text.isEmpty else { throw AppleSpeechError.noSpeech }
        return text
    }

    static func readAudio(_ url: URL) throws -> [Float] {
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0, size <= ResourcePolicy.maximumAudioBytes else { throw AppleSpeechError.invalidAudio }
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard (8_000...48_000).contains(source.sampleRate), source.channelCount <= 2,
              file.length > 0, Double(file.length) / source.sampleRate <= 60.25,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source, to: target),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 964_000)
        else { throw AppleSpeechError.invalidAudio }
        var conversionError: NSError?, inputError: Error?
        var currentInput: AVAudioPCMBuffer?
        let state = converter.convert(to: output, error: &conversionError) { requested, status in
            guard file.framePosition < file.length, requested > 0 else { status.pointee = .endOfStream; return nil }
            // Core Audio may request less than the file length. Returning the
            // entire recording once can discard unconsumed tail frames.
            let frames = min(requested, 65_536, AVAudioPacketCount(file.length - file.framePosition))
            guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames) else {
                inputError = AppleSpeechError.invalidAudio; status.pointee = .endOfStream; return nil
            }
            do { try file.read(into: input, frameCount: frames) }
            catch { inputError = error; status.pointee = .endOfStream; return nil }
            currentInput = input
            status.pointee = input.frameLength > 0 ? .haveData : .endOfStream
            return input.frameLength > 0 ? currentInput : nil
        }
        guard state != .error, conversionError == nil, inputError == nil, output.frameLength > 0,
              let channel = output.floatChannelData?[0] else { throw AppleSpeechError.invalidAudio }
        let result = Array(UnsafeBufferPointer(start: channel, count: min(Int(output.frameLength), 960_000)))
        guard result.allSatisfy(\.isFinite), result.contains(where: { abs($0) > 0.00001 }) else { throw AppleSpeechError.noSpeech }
        return result
    }

    static func tokenIDs(_ logits: MLMultiArray, validSamples: Int) throws -> [Int] {
        let shape = logits.shape.map(\.intValue), strides = logits.strides.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, (1...ResourcePolicy.onDeviceMaximumFrames).contains(shape[1]),
              shape[2] == ResourcePolicy.onDeviceVocabularyCount else { throw AppleSpeechError.invalidModel }
        // Wav2vec2's convolution frontend has 400-sample receptive field and
        // 320-sample stride. Ignore frames belonging only to zero padding.
        let frames = min(shape[1], max(1, (validSamples - 400) / 320 + 1)), vocabulary = shape[2]
        var ids: [Int] = []; ids.reserveCapacity(frames)
        let contiguous = strides[2] == 1 && strides[1] == vocabulary
        for frame in 0..<frames {
            try Task.checkCancellation()
            var best = -Float.infinity, index = 0
            for token in 0..<vocabulary {
                let value: Float
                if contiguous && logits.dataType == .float32 {
                    value = logits.dataPointer.assumingMemoryBound(to: Float.self)[frame * vocabulary + token]
                } else if contiguous && logits.dataType == .float16 {
                    value = Float(logits.dataPointer.assumingMemoryBound(to: Float16.self)[frame * vocabulary + token])
                } else {
                    value = logits[[0, NSNumber(value: frame), NSNumber(value: token)]].floatValue
                }
                guard value.isFinite else { throw AppleSpeechError.invalidModel }
                if value > best { best = value; index = token }
            }
            ids.append(index)
        }
        return ids
    }
}
