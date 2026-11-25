import Foundation
import SQLite3

/// Used only by ConversationArchive's actor. No view owns a database connection.
final class ConversationDatabase {
    private var handle: OpaquePointer?
    private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    enum Value { case text(String), number(Double), integer(Int), null }

    init(url: URL) throws {
        guard sqlite3_open_v2(url.path, &handle, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
            let detail = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "Cannot open database"
            if let handle { sqlite3_close(handle) }; handle = nil
            throw ArchiveError.database(detail)
        }
        do {
            let version = try statement("PRAGMA user_version")
            let versionCode = sqlite3_step(version)
            let schema = sqlite3_column_int(version, 0)
            sqlite3_finalize(version)
            guard versionCode == SQLITE_ROW, schema <= 2 else { throw ArchiveError.unavailable }
            try execute("PRAGMA auto_vacuum=INCREMENTAL")
            try execute("PRAGMA foreign_keys=ON")
            try execute("PRAGMA journal_mode=WAL")
            try execute("PRAGMA synchronous=FULL")
            try execute("PRAGMA cache_size=-2048")
            try execute("PRAGMA mmap_size=0")
            try execute("PRAGMA temp_store=FILE")
            try execute("PRAGMA secure_delete=ON")
            sqlite3_limit(handle, SQLITE_LIMIT_LENGTH, 200_000)
            sqlite3_busy_timeout(handle, 2_000)
            try execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            try execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL, title TEXT NOT NULL DEFAULT '', message_count INTEGER NOT NULL DEFAULT 0, first_user_created REAL, first_user_id TEXT)")
            try execute("CREATE TABLE IF NOT EXISTS messages (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, role TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL, preview INTEGER NOT NULL, audio TEXT)")
            try execute("CREATE INDEX IF NOT EXISTS conversation_order ON conversations(updated DESC, id DESC)")
            try execute("CREATE INDEX IF NOT EXISTS message_order ON messages(conversation_id, created DESC, sequence DESC)")
            try execute("CREATE INDEX IF NOT EXISTS audio_reference ON messages(audio) WHERE audio IS NOT NULL")
            // Foundation's Codable dates use its reference epoch. Keeping that
            // epoch avoids precision loss from adding/subtracting the Unix offset.
            try transaction {
                if schema == 1 {
                    try execute("UPDATE conversations SET created=created-978307200,updated=updated-978307200,first_user_created=first_user_created-978307200")
                    try execute("UPDATE messages SET created=created-978307200")
                }
                try execute("PRAGMA user_version=2")
            }
        } catch { sqlite3_close(handle); handle = nil; throw error }
    }

    deinit { if let handle { sqlite3_close(handle) } }

    private func statement(_ sql: String, _ values: [Value] = []) throws -> OpaquePointer {
        var result: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &result, nil) == SQLITE_OK, let result else { throw failure() }
        for (offset, value) in values.enumerated() {
            let index = Int32(offset + 1)
            let code: Int32
            switch value {
            case .text(let text): code = text.withCString { sqlite3_bind_text(result, index, $0, Int32(text.utf8.count), transient) }
            case .number(let value): code = sqlite3_bind_double(result, index, value)
            case .integer(let value): code = sqlite3_bind_int64(result, index, Int64(value))
            case .null: code = sqlite3_bind_null(result, index)
            }
            guard code == SQLITE_OK else { sqlite3_finalize(result); throw failure() }
        }
        return result
    }
    private func failure() -> ArchiveError { .database(handle.map { String(cString: sqlite3_errmsg($0)) } ?? "Database closed") }
    private func text(_ row: OpaquePointer, _ column: Int32) -> String {
        guard let bytes = sqlite3_column_text(row, column) else { return "" }
        return String(decoding: UnsafeBufferPointer(start: bytes, count: Int(sqlite3_column_bytes(row, column))), as: UTF8.self)
    }
    private func execute(_ sql: String, _ values: [Value] = []) throws {
        let query = try statement(sql, values); defer { sqlite3_finalize(query) }
        var code = sqlite3_step(query)
        while code == SQLITE_ROW { code = sqlite3_step(query) }
        guard code == SQLITE_DONE else { throw failure() }
    }
    private func transaction(_ work: () throws -> Void) throws {
        try execute("BEGIN IMMEDIATE")
        do { try work(); try execute("COMMIT") }
        catch { try? execute("ROLLBACK"); throw error }
    }

    func migrateLegacy(from url: URL) throws {
        let marker = try statement("SELECT value FROM metadata WHERE key='legacy_imported'")
        let markerCode = sqlite3_step(marker)
        let imported = markerCode == SQLITE_ROW
        sqlite3_finalize(marker)
        guard markerCode == SQLITE_ROW || markerCode == SQLITE_DONE else { throw failure() }
        if imported {
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
            return
        }
        try transaction {
            if FileManager.default.fileExists(atPath: url.path) {
                try LegacyConversationReader.forEach(at: url) { conversation in
                    for message in conversation.messages {
                        try insert(message, conversationID: conversation.id, createdAt: conversation.createdAt, preserveLegacyAudio: true)
                    }
                }
            }
            try execute("INSERT INTO metadata(key,value) VALUES('legacy_imported','1')")
        }
        // Only retire JSON after the whole import and its marker are committed.
        if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        try? execute("PRAGMA wal_checkpoint(TRUNCATE)")
    }

    private func insert(_ message: Message, conversationID: UUID, createdAt: Date, preserveLegacyAudio: Bool) throws {
        let existing = try statement("SELECT conversation_id FROM messages WHERE id=?", [.text(message.id.uuidString)])
        let existingCode = sqlite3_step(existing)
        let isNew = existingCode == SQLITE_DONE
        let sameConversation = existingCode == SQLITE_ROW && text(existing, 0) == conversationID.uuidString
        sqlite3_finalize(existing)
        guard isNew || sameConversation else { throw ArchiveError.invalidLegacyHistory }
        try execute("INSERT OR IGNORE INTO conversations(id,created,updated) VALUES(?,?,?)", [.text(conversationID.uuidString), .number(createdAt.timeIntervalSinceReferenceDate), .number(message.createdAt.timeIntervalSinceReferenceDate)])
        let audio: Value = preserveLegacyAudio ? message.audioFilename.map { .text(URL(fileURLWithPath: $0).lastPathComponent) } ?? .null : .null
        try execute("INSERT INTO messages(id,conversation_id,role,text,created,preview,audio) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text,preview=excluded.preview", [.text(message.id.uuidString), .text(conversationID.uuidString), .text(message.role.rawValue), .text(message.text), .number(message.createdAt.timeIntervalSinceReferenceDate), .integer(message.isPreview ? 1 : 0), audio])
        try execute("UPDATE conversations SET updated=MAX(updated,?) WHERE id=?", [.number(message.createdAt.timeIntervalSinceReferenceDate), .text(conversationID.uuidString)])
        if isNew { try execute("UPDATE conversations SET message_count=message_count+1 WHERE id=?", [.text(conversationID.uuidString)]) }
        if message.role == .user {
            try execute("UPDATE conversations SET title=?,first_user_created=?,first_user_id=? WHERE id=? AND (first_user_created IS NULL OR first_user_created>? OR first_user_id=?)", [.text(String(message.text.prefix(65))), .number(message.createdAt.timeIntervalSinceReferenceDate), .text(message.id.uuidString), .text(conversationID.uuidString), .number(message.createdAt.timeIntervalSinceReferenceDate), .text(message.id.uuidString)])
        }
    }
    func save(_ message: Message, conversationID: UUID, createdAt: Date) throws {
        try transaction { try insert(message, conversationID: conversationID, createdAt: createdAt, preserveLegacyAudio: false) }
    }
    func save(_ messages: [Message], conversationID: UUID, createdAt: Date) throws {
        try transaction { for message in messages where !message.isVoicePlaceholder { try insert(message, conversationID: conversationID, createdAt: createdAt, preserveLegacyAudio: false) } }
    }

    func summaries(before: ConversationSummary? = nil, after: ConversationSummary? = nil) throws -> ArchivePage<ConversationSummary> {
        var values: [Value] = [], condition = ""
        if let cursor = before ?? after {
            let op = before != nil ? "<" : ">"
            condition = " WHERE (c.updated \(op) ? OR (c.updated=? AND c.id \(op) ?))"
            values = [.number(cursor.updatedAt.timeIntervalSinceReferenceDate), .number(cursor.updatedAt.timeIntervalSinceReferenceDate), .text(cursor.id.uuidString)]
        }
        let direction = after == nil ? "DESC" : "ASC"
        let sql = "SELECT c.id,c.created,c.updated,c.title,c.message_count FROM conversations c\(condition) ORDER BY c.updated \(direction),c.id \(direction) LIMIT \(ResourcePolicy.historyPageSize)"
        let query = try statement(sql, values); defer { sqlite3_finalize(query) }
        var items: [ConversationSummary] = []
        var code = sqlite3_step(query)
        while code == SQLITE_ROW {
            guard let id = UUID(uuidString: text(query, 0)) else { throw ArchiveError.unavailable }
            items.append(.init(id: id, createdAt: Date(timeIntervalSinceReferenceDate: sqlite3_column_double(query, 1)), updatedAt: Date(timeIntervalSinceReferenceDate: sqlite3_column_double(query, 2)), firstUserText: text(query, 3), messageCount: Int(sqlite3_column_int64(query, 4))))
            code = sqlite3_step(query)
        }
        guard code == SQLITE_DONE else { throw failure() }
        if after != nil { items.reverse() }
        guard let first = items.first, let last = items.last else { return .init(items: [], hasOlder: false, hasNewer: false) }
        return try .init(items: items, hasOlder: summaryExists(relativeTo: last, older: true), hasNewer: summaryExists(relativeTo: first, older: false))
    }
    private func summaryExists(relativeTo cursor: ConversationSummary, older: Bool) throws -> Bool {
        let op = older ? "<" : ">"
        let query = try statement("SELECT 1 FROM conversations WHERE updated \(op) ? OR (updated=? AND id \(op) ?) LIMIT 1", [.number(cursor.updatedAt.timeIntervalSinceReferenceDate), .number(cursor.updatedAt.timeIntervalSinceReferenceDate), .text(cursor.id.uuidString)])
        defer { sqlite3_finalize(query) }; return try exists(query)
    }

    private func cursor(for id: UUID) throws -> (Double, Int)? {
        let query = try statement("SELECT created,sequence FROM messages WHERE id=?", [.text(id.uuidString)])
        defer { sqlite3_finalize(query) }
        return try exists(query) ? (sqlite3_column_double(query, 0), Int(sqlite3_column_int64(query, 1))) : nil
    }
    func messages(in conversationID: UUID, before: UUID? = nil, after: UUID? = nil, contextOnly: Bool = false) throws -> ArchivePage<Message> {
        var values: [Value] = [.text(conversationID.uuidString)]
        var condition = ""
        if let id = before ?? after, let (date, sequence) = try cursor(for: id) {
            let op = before != nil ? "<" : ">"
            condition = " AND (created \(op) ? OR (created=? AND sequence \(op) ?))"
            values += [.number(date), .number(date), .integer(sequence)]
        }
        if contextOnly { condition += " AND preview=0 AND (audio IS NULL OR text NOT LIKE 'Voice message ·%')" }
        let direction = after == nil ? "DESC" : "ASC"
        let limit = contextOnly ? ResourcePolicy.contextMessageCount + 1 : ResourcePolicy.messagePageSize
        let query = try statement("SELECT id,role,text,created,preview,audio FROM messages WHERE conversation_id=?\(condition) ORDER BY created \(direction),sequence \(direction) LIMIT \(limit)", values)
        defer { sqlite3_finalize(query) }
        var items: [Message] = []
        var code = sqlite3_step(query)
        while code == SQLITE_ROW {
            guard let id = UUID(uuidString: text(query, 0)), let role = Message.Role(rawValue: text(query, 1)) else { throw ArchiveError.unavailable }
            items.append(.init(id: id, role: role, text: text(query, 2), createdAt: Date(timeIntervalSinceReferenceDate: sqlite3_column_double(query, 3)), isPreview: sqlite3_column_int(query, 4) != 0, audioFilename: sqlite3_column_type(query, 5) == SQLITE_NULL ? nil : text(query, 5)))
            code = sqlite3_step(query)
        }
        guard code == SQLITE_DONE else { throw failure() }
        if after == nil { items.reverse() }
        guard !contextOnly, let first = items.first, let last = items.last else { return .init(items: items, hasOlder: false, hasNewer: false) }
        return try .init(items: items, hasOlder: messageExists(in: conversationID, relativeTo: first.id, older: true), hasNewer: messageExists(in: conversationID, relativeTo: last.id, older: false))
    }
    private func messageExists(in conversationID: UUID, relativeTo id: UUID, older: Bool) throws -> Bool {
        guard let (date, sequence) = try cursor(for: id) else { return false }
        let op = older ? "<" : ">"
        let query = try statement("SELECT 1 FROM messages WHERE conversation_id=? AND (created \(op) ? OR (created=? AND sequence \(op) ?)) LIMIT 1", [.text(conversationID.uuidString), .number(date), .number(date), .integer(sequence)])
        defer { sqlite3_finalize(query) }; return try exists(query)
    }
    private func exists(_ query: OpaquePointer) throws -> Bool {
        let code = sqlite3_step(query)
        guard code == SQLITE_ROW || code == SQLITE_DONE else { throw failure() }
        return code == SQLITE_ROW
    }
    func referencesAudio(_ filename: String) throws -> Bool {
        let query = try statement("SELECT 1 FROM messages WHERE audio=? LIMIT 1", [.text(filename)])
        defer { sqlite3_finalize(query) }
        let code = sqlite3_step(query)
        guard code == SQLITE_ROW || code == SQLITE_DONE else { throw failure() }
        return code == SQLITE_ROW
    }
    func deleteConversation(_ id: UUID) throws {
        try transaction { try execute("DELETE FROM conversations WHERE id=?", [.text(id.uuidString)]) }
        try execute("PRAGMA incremental_vacuum(128)")
        try execute("PRAGMA wal_checkpoint(TRUNCATE)")
    }
}
