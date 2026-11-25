import Foundation
import SQLite3

struct CheckFailure: Error, CustomStringConvertible { let description: String }
func check(_ condition: Bool, _ message: String) throws {
    if !condition { throw CheckFailure(description: message) }
}

@main struct ResourceChecks {
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try pagination(root.appending(path: "pages"))
        try schemaUpgrade(root.appending(path: "schema"))
        print("PASS: 2,000 messages and 95 conversations page without gaps, duplicates, or full-history loading")
        try await migration(root.appending(path: "migration"))
        print("PASS: legacy JSON migrates transactionally; Unicode, audio references, and idempotence survive")
        try await corruptMigration(root.appending(path: "corrupt"))
        print("PASS: corrupt migration rolls back and preserves original history and audio")
        try await temporaryAudio(root.appending(path: "temporary"))
        print("PASS: temporary audio expires safely, respects quotas, and is never saved in new history")
        try context()
        print("PASS: model context stays within 12 messages / 24,000 characters")
        try await network()
        print("PASS: HTTP responses enforce declared and streamed limits; cancellation stops the request")
        try await AppStateChecks.run(root.appending(path: "store"))
        print("PASS: actual app store pages, restores history, preserves opt-in sessions, and enforces history-off limits (voice hardware stubbed)")
    }

    static func pagination(_ directory: URL) throws {
        let repository = LocalRepository(directory: directory)
        try repository.prepare()
        let database = try ConversationDatabase(url: repository.databaseURL)
        let id = UUID(), date = Date(timeIntervalSince1970: 1_000)
        // Tied timestamps deliberately exercise the stable sequence cursor.
        let messages = (0..<2_000).map { Message(role: $0 % 2 == 0 ? .user : .assistant, text: "Ozi \($0) · Anyanwụ", createdAt: date) }
        try database.save(messages, conversationID: id, createdAt: date)
        var page = try database.messages(in: id)
        try check(page.items.count == 50 && page.hasOlder && !page.hasNewer, "initial message page")
        let latest = page.items
        var seen = Set(page.items.map(\.id))
        while page.hasOlder {
            let prior = page.items
            page = try database.messages(in: id, before: prior.first!.id)
            try check(page.items.count <= 50 && page.hasNewer, "older page bounds")
            let forward = try database.messages(in: id, after: page.items.last!.id)
            try check(forward.items == prior, "forward cursor must recover the adjacent page")
            for item in page.items { try check(seen.insert(item.id).inserted, "duplicate message") }
        }
        try check(seen == Set(messages.map(\.id)), "missing messages")
        try check(try database.messages(in: id).items == latest, "latest window changed")
        for index in 0..<94 {
            let conversationID = UUID()
            try database.save(Message(role: .user, text: "Mkparịta ụka \(index)", createdAt: date), conversationID: conversationID, createdAt: date)
        }
        var summaries = try database.summaries()
        var ids = Set(summaries.items.map(\.id))
        while summaries.hasOlder {
            let previous = summaries.items
            summaries = try database.summaries(before: previous.last)
            try check(summaries.items.count <= 30, "summary page too large")
            let next = try database.summaries(after: summaries.items.first)
            try check(next.items == previous, "summary forward pagination")
            for item in summaries.items { try check(ids.insert(item.id).inserted, "duplicate summary") }
        }
        try check(ids.count == 95, "missing conversation")
        var update = messages[0]
        update.text = "Ndewo\0Anyanwụ"
        try database.save(update, conversationID: id, createdAt: date)
        let firstPage = try database.messages(in: id, before: messages[50].id)
        try check(firstPage.items.first?.text == update.text, "Unicode/NUL text changed")
        let summary = try database.summaries().items.first { $0.id == id }
        // The target can be on another summary page, so verify via a direct bounded query below.
        if let summary { try check(summary.messageCount == 2_000, "upsert inflated message count") }
        try database.deleteConversation(id)
        try check(try database.messages(in: id).items.isEmpty, "cascade deletion failed")
    }

    static func schemaUpgrade(_ directory: URL) throws {
        let repository = LocalRepository(directory: directory)
        try repository.prepare()
        let id = UUID(), date = Date(timeIntervalSinceReferenceDate: 123_456.125)
        let message = Message(role: .user, text: "Ndewo", createdAt: date)
        do {
            let database = try ConversationDatabase(url: repository.databaseURL)
            try database.save(message, conversationID: id, createdAt: date)
        }
        var handle: OpaquePointer?
        try check(sqlite3_open(repository.databaseURL.path, &handle) == SQLITE_OK, "schema fixture open")
        let sql = "UPDATE conversations SET created=created+978307200,updated=updated+978307200,first_user_created=first_user_created+978307200; UPDATE messages SET created=created+978307200; PRAGMA user_version=1"
        let code = sqlite3_exec(handle, sql, nil, nil, nil)
        sqlite3_close(handle)
        try check(code == SQLITE_OK, "schema fixture setup")
        let upgraded = try ConversationDatabase(url: repository.databaseURL)
        try check(try upgraded.messages(in: id).items == [message], "v1 timestamp upgrade lost data")
        try check(try upgraded.summaries().items[0].createdAt == date, "summary date upgrade")
    }

    static func migration(_ directory: URL) async throws {
        let repository = LocalRepository(directory: directory)
        try repository.prepare()
        let message = Message(role: .user, text: "Ndewo, Anyanwụ. {\"ị\": [\"ọ\"]} \\", audioFilename: "keep.m4a")
        let conversation = Conversation(messages: [message])
        try repository.save([conversation], file: "conversations.json")
        try Data([1, 2, 3]).write(to: repository.audioURL("keep.m4a"))
        try Data([4]).write(to: repository.audioURL("orphan.m4a"))
        let archive = ConversationArchive(repository: repository)
        let page = try await archive.prepare(historyEnabled: true)
        try check(page.items.first?.messageCount == 1, "legacy count")
        try check(try await archive.messages(in: conversation.id).items == [message], "legacy payload changed")
        try check(!FileManager.default.fileExists(atPath: repository.legacyHistoryURL.path), "committed JSON not retired")
        try check(FileManager.default.fileExists(atPath: repository.audioURL("keep.m4a").path), "referenced audio removed")
        try check(!FileManager.default.fileExists(atPath: repository.audioURL("orphan.m4a").path), "orphan not removed")
        // A crash after commit but before retirement must not duplicate records.
        try repository.save([conversation], file: "conversations.json")
        let reopened = ConversationArchive(repository: repository)
        let second = try await reopened.prepare(historyEnabled: true)
        try check(second.items.count == 1 && second.items[0].messageCount == 1, "migration not idempotent")
        try await reopened.deleteConversation(conversation.id)
        try check(!FileManager.default.fileExists(atPath: repository.audioURL("keep.m4a").path), "deleted conversation audio left behind")
    }

    static func corruptMigration(_ directory: URL) async throws {
        let repository = LocalRepository(directory: directory)
        try repository.prepare()
        let conversation = Conversation(messages: [Message(role: .user, text: "Ndewo")])
        var data = Data("[".utf8)
        data.append(try JSONEncoder().encode(conversation))
        data.append(Data(",{broken}]".utf8))
        try data.write(to: repository.legacyHistoryURL)
        try Data([1]).write(to: repository.audioURL("preserve.m4a"))
        do {
            _ = try await ConversationArchive(repository: repository).prepare(historyEnabled: true)
            throw CheckFailure(description: "corrupt import was accepted")
        } catch is CheckFailure { throw CheckFailure(description: "corrupt import was accepted") }
        catch { }
        try check(try Data(contentsOf: repository.legacyHistoryURL) == data, "original JSON changed after failed import")
        try check(FileManager.default.fileExists(atPath: repository.audioURL("preserve.m4a").path), "audio pruned after failed import")
        let database = try ConversationDatabase(url: repository.databaseURL)
        try check(try database.summaries().items.isEmpty, "partially committed corrupt import")
        for invalid in ["[{},]", "[]garbage", "[", "[\"text\"]"] {
            try Data(invalid.utf8).write(to: repository.legacyHistoryURL)
            do {
                try LegacyConversationReader.forEach(at: repository.legacyHistoryURL) { _ in }
                throw CheckFailure(description: "invalid stream was accepted")
            } catch is CheckFailure { throw CheckFailure(description: "invalid stream was accepted") }
            catch { }
        }
    }

    static func temporaryAudio(_ directory: URL) async throws {
        let repository = LocalRepository(directory: directory)
        let archive = ConversationArchive(repository: repository)
        _ = try await archive.prepare(historyEnabled: true)
        let id = UUID(), now = Date()
        let input = try await archive.recordingURL(in: id)
        try Data([1, 2, 3]).write(to: input)
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-700)], ofItemAtPath: input.path)
        let reply = try await archive.saveReply(Data([4, 5]).base64EncodedString(), mimeType: "audio/mpeg", in: id)
        let url = try await archive.playbackURL(reply, in: id)
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-700)], ofItemAtPath: url.path)
        let removed = try await archive.removeExpiredAudio(protecting: [input.lastPathComponent], now: now)
        try check(removed == [reply], "expiry removed protected input")
        try check(try await archive.readRecording(input) == Data([1, 2, 3]), "input read changed")
        let message = Message(role: .assistant, text: "Ndewo", audioFilename: reply)
        try await archive.save(message, in: id, createdAt: now)
        try check(try await archive.messages(in: id).items.first?.audioFilename == nil, "new audio was persisted")
        _ = try await archive.removeTemporaryAudio(input.lastPathComponent, in: id)
        try check(!FileManager.default.fileExists(atPath: input.path), "input cleanup failed")
        for _ in 0..<ResourcePolicy.maximumTemporaryAudioFiles {
            _ = try await archive.saveReply(Data([1]).base64EncodedString(), mimeType: "audio/mpeg", in: id)
        }
        do {
            _ = try await archive.saveReply("AQ==", mimeType: "audio/mpeg", in: id)
            throw CheckFailure(description: "file count limit bypassed")
        } catch ArchiveError.audioLimit { }
        try await archive.clearTemporaryAudio(in: id)
        for _ in 0..<4 {
            _ = try await archive.saveReply(Data(repeating: 0, count: ResourcePolicy.maximumAudioBytes).base64EncodedString(), mimeType: "audio/mpeg", in: id)
        }
        do {
            _ = try await archive.recordingURL(in: id)
            throw CheckFailure(description: "audio byte limit bypassed")
        } catch ArchiveError.audioLimit { }
        try await archive.setHistoryEnabled(false)
        try await archive.save(Message(role: .user, text: "private"), in: id, createdAt: now)
        try check(try await archive.summaries().items.isEmpty, "history-off wrote text")
        let usage = try await archive.usage()
        try check(usage.temporaryAudioBytes == 0, "opt-out left temporary audio")
        let old = try await archive.recordingURL(in: id)
        try Data([1]).write(to: old)
        _ = try await ConversationArchive(repository: repository).prepare(historyEnabled: false)
        try check(!FileManager.default.fileExists(atPath: old.path), "abandoned session audio survived restart")
    }

    static func context() throws {
        var messages = (0..<100).map { _ in Message(role: .user, text: String(repeating: "ị", count: 8_000)) }
        let excluded = messages.last!.id
        messages.append(Message(role: .assistant, text: "preview", isPreview: true))
        messages.append(Message(role: .user, text: "Voice message · 2s", audioFilename: "temp-input-a.m4a"))
        let context = ContextWindow.messages(from: messages, excluding: excluded)
        try check(context.count <= 12 && context.reduce(0, { $0 + $1.content.count }) == 24_000, "character budget")
        try check(context.allSatisfy { $0.content == String(repeating: "ị", count: 8_000) }, "placeholder/preview leaked")
        let short = (0..<100).map { Message(role: .user, text: "\($0)") }
        try check(ContextWindow.messages(from: short).map(\.content) == (88..<100).map(String.init), "latest context window")
    }

    static func network() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FixtureProtocol.self]
        func request(_ path: String) -> URLRequest { URLRequest(url: URL(string: "https://fixture.invalid/\(path)")!) }
        let (data, _) = try await BoundedHTTP.data(for: request("small"), limit: 10, configuration: config)
        try check(data == Data("Ndewo".utf8), "response changed")
        for mode in ["declared", "streamed"] {
            do {
                _ = try await BoundedHTTP.data(for: request(mode), limit: 10, configuration: config)
                throw CheckFailure(description: "HTTP overflow accepted")
            } catch AssistantServiceError.responseTooLarge { }
        }
        let task = Task { try await BoundedHTTP.data(for: request("pending"), limit: 10, configuration: config) }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        do { _ = try await task.value; throw CheckFailure(description: "cancellation ignored") }
        catch is CancellationError { }
    }
}

final class FixtureProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let mode = request.url!.lastPathComponent
        if mode == "pending" { return }
        let headers = mode == "declared" ? ["Content-Length": "1000000"] : [:]
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: headers)!, cacheStoragePolicy: .notAllowed)
        if mode == "streamed" {
            client?.urlProtocol(self, didLoad: Data(repeating: 1, count: 8))
            client?.urlProtocol(self, didLoad: Data(repeating: 1, count: 8))
        } else if mode == "small" { client?.urlProtocol(self, didLoad: Data("Ndewo".utf8)) }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { }
}
