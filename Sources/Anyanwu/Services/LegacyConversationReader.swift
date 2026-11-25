import Foundation

/// Bounded, one-time reader for the original top-level JSON conversation array.
/// One record is decoded at a time; malformed/oversized records abort migration.
struct LegacyConversationReader {
    static func forEach(at url: URL, _ consume: (Conversation) throws -> Void) throws {
        guard let stream = InputStream(url: url) else { throw ArchiveError.invalidLegacyHistory }
        stream.open()
        defer { stream.close() }
        var buffer = [UInt8](repeating: 0, count: 32_768)
        var started = false, ended = false, inString = false, escaped = false
        var depth = 0, expectingRecord = true, mayEnd = true
        var record = Data()
        while true {
            let count = stream.read(&buffer, maxLength: 32_768)
            guard count >= 0 else { throw ArchiveError.invalidLegacyHistory }
            if count == 0 { break }
            for byte in buffer.prefix(count) {
                if depth == 0 {
                    if [9, 10, 13, 32].contains(byte) { continue }
                    if !started {
                        guard byte == 91 else { throw ArchiveError.invalidLegacyHistory }
                        started = true
                    } else if ended { throw ArchiveError.invalidLegacyHistory }
                    else if expectingRecord {
                        if byte == 93 && mayEnd { ended = true; continue }
                        guard byte == 123 else { throw ArchiveError.invalidLegacyHistory }
                        record = Data([byte]); depth = 1; inString = false; escaped = false
                    } else if byte == 44 { expectingRecord = true; mayEnd = false }
                    else if byte == 93 { ended = true }
                    else { throw ArchiveError.invalidLegacyHistory }
                    continue
                }
                record.append(byte)
                guard record.count <= 4_000_000 else { throw ArchiveError.invalidLegacyHistory }
                if inString {
                    if escaped { escaped = false }
                    else if byte == 92 { escaped = true }
                    else if byte == 34 { inString = false }
                } else if byte == 34 { inString = true }
                else if byte == 123 || byte == 91 { depth += 1 }
                else if byte == 125 || byte == 93 { depth -= 1 }
                if depth == 0 {
                    try Task.checkCancellation()
                    let conversation = try JSONDecoder().decode(Conversation.self, from: record)
                    try consume(conversation)
                    record.removeAll(keepingCapacity: false)
                    expectingRecord = false; mayEnd = true
                }
            }
        }
        guard started, ended, depth == 0, !inString else { throw ArchiveError.invalidLegacyHistory }
    }
}
