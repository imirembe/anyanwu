import Foundation

/// Reads only the bounded piece table of SentencePiece's public protobuf wire
/// format. No third-party tokenizer runtime or remote executable code is used.
struct SentencePieceLexicon: Sendable {
    struct Piece: Sendable { let text: String; let kind: Int }
    let pieces: [Piece]

    init(data: Data) throws {
        guard !data.isEmpty, data.count <= 1_000_000 else { throw AppleSpeechError.invalidModel }
        var reader = WireReader(bytes: Array(data))
        var result: [Piece] = []
        while !reader.finished {
            let tag = try reader.varint()
            if tag == 10 { // ModelProto.pieces, a repeated length-delimited message.
                var part = WireReader(bytes: try reader.block())
                var text: String?, kind = 1
                while !part.finished {
                    let field = try part.varint()
                    if field == 10 {
                        let bytes = try part.block()
                        guard bytes.count <= 128, let value = String(bytes: bytes, encoding: .utf8) else { throw AppleSpeechError.invalidModel }
                        text = value
                    } else if field == 24 { kind = try part.varint() }
                    else { try part.skip(wire: field & 7) }
                }
                guard let text, result.count < ResourcePolicy.onDeviceVocabularyCount else { throw AppleSpeechError.invalidModel }
                result.append(Piece(text: text, kind: kind))
            } else { try reader.skip(wire: tag & 7) }
        }
        guard result.count == ResourcePolicy.onDeviceVocabularyCount else { throw AppleSpeechError.invalidModel }
        pieces = result
    }

    func decode(_ ids: [Int]) throws -> String {
        var text = ""
        for id in ids {
            guard pieces.indices.contains(id) else { throw AppleSpeechError.invalidModel }
            // BOS=0, CTC blank/PAD=1, EOS=2, UNK=3; also omit protobuf
            // unknown/control/unused/byte pieces, matching the upstream decoder.
            guard id > 3, [1, 4].contains(pieces[id].kind) else { continue }
            text += pieces[id].text
            guard text.utf8.count <= 100_000 else { throw AppleSpeechError.invalidModel }
        }
        return text.replacingOccurrences(of: "\u{2581}", with: " ")
            .precomposedStringWithCanonicalMapping.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private struct WireReader {
    let bytes: [UInt8]
    var index = 0
    var finished: Bool { index == bytes.count }
    mutating func varint() throws -> Int {
        var value: UInt64 = 0
        for shift in stride(from: 0, to: 63, by: 7) {
            guard index < bytes.count else { throw AppleSpeechError.invalidModel }
            let byte = bytes[index]; index += 1
            value |= UInt64(byte & 127) << shift
            if byte & 128 == 0 {
                guard value <= UInt64(Int.max) else { throw AppleSpeechError.invalidModel }
                return Int(value)
            }
        }
        throw AppleSpeechError.invalidModel
    }
    mutating func block() throws -> [UInt8] {
        let count = try varint()
        guard count <= bytes.count - index else { throw AppleSpeechError.invalidModel }
        defer { index += count }
        return Array(bytes[index..<index + count])
    }
    mutating func skip(wire: Int) throws {
        let count: Int
        switch wire {
        case 0: _ = try varint(); return
        case 1: count = 8
        case 2: count = try varint()
        case 5: count = 4
        default: throw AppleSpeechError.invalidModel
        }
        guard count <= bytes.count - index else { throw AppleSpeechError.invalidModel }
        index += count
    }
}

enum AppleCTC {
    static func collapse(_ ids: [Int]) -> [Int] {
        var result: [Int] = [], previous: Int?
        for id in ids {
            if id != previous { result.append(id) }
            previous = id // Blanks reset repetition, so a blank-separated repeat survives.
        }
        return result
    }

    static func normalizedWindow(_ samples: ArraySlice<Float>) throws -> [Float] {
        guard !samples.isEmpty, samples.count <= ResourcePolicy.onDeviceInputSamples,
              samples.allSatisfy(\.isFinite) else { throw AppleSpeechError.invalidAudio }
        let mean = samples.reduce(0.0) { $0 + Double($1) } / Double(samples.count)
        let variance = samples.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / Double(samples.count)
        let scale = 1 / sqrt(variance + 0.00001)
        var output = samples.map { Float((Double($0) - mean) * scale) }
        output.append(contentsOf: repeatElement(0, count: ResourcePolicy.onDeviceInputSamples - output.count))
        return output
    }
}
