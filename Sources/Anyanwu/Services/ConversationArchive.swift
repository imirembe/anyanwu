import Foundation

/// Serial worker owns disk I/O, SQLite, and all disposable audio. UI keeps pages only.
actor ConversationArchive {
    let repository: LocalRepository
    private var database: ConversationDatabase?
    private var historyEnabled = true
    private let sessionID = "session-" + UUID().uuidString
    private var sessionDirectory: URL { repository.temporaryAudioDirectory.appending(path: sessionID, directoryHint: .isDirectory) }

    init(repository: LocalRepository) { self.repository = repository }

    func prepare(historyEnabled: Bool) throws -> ArchivePage<ConversationSummary> {
        try repository.prepare()
        try clearOldTemporarySessions()
        try FileManager.default.createDirectory(at: sessionDirectory, withIntermediateDirectories: true)
        self.historyEnabled = historyEnabled
        if !historyEnabled { try eraseHistory() }
        else {
            let opened = try ConversationDatabase(url: repository.databaseURL)
            try opened.migrateLegacy(from: repository.legacyHistoryURL)
            database = opened
            try pruneLegacyOrphans()
        }
        return try summaries()
    }

    func saveSettings(_ settings: AssistantSettings) throws { try repository.save(settings, file: "settings.json") }

    func setHistoryEnabled(_ enabled: Bool) throws {
        if !enabled && historyEnabled { try eraseHistory() }
        historyEnabled = enabled
        if enabled && database == nil {
            let opened = try ConversationDatabase(url: repository.databaseURL)
            try opened.migrateLegacy(from: repository.legacyHistoryURL)
            database = opened
        }
    }

    private func opened() throws -> ConversationDatabase {
        guard let database else { throw ArchiveError.unavailable }
        return database
    }

    func save(_ message: Message, in id: UUID, createdAt: Date) throws {
        try Task.checkCancellation()
        guard historyEnabled, !message.isVoicePlaceholder else { return }
        try opened().save(message, conversationID: id, createdAt: createdAt)
    }

    func saveSession(_ messages: [Message], in id: UUID, createdAt: Date) throws {
        try Task.checkCancellation()
        guard historyEnabled else { return }
        try opened().save(messages, conversationID: id, createdAt: createdAt)
    }

    func summaries(before: ConversationSummary? = nil, after: ConversationSummary? = nil) throws -> ArchivePage<ConversationSummary> {
        guard historyEnabled else { return .init(items: [], hasOlder: false, hasNewer: false) }
        return try opened().summaries(before: before, after: after)
    }

    func messages(in id: UUID, before: UUID? = nil, after: UUID? = nil) throws -> ArchivePage<Message> {
        guard historyEnabled else { return .init(items: [], hasOlder: false, hasNewer: false) }
        return try opened().messages(in: id, before: before, after: after)
    }

    func context(in id: UUID, excluding messageID: UUID) throws -> [TurnRequest.ContextMessage] {
        guard historyEnabled else { return [] }
        let page = try opened().messages(in: id, contextOnly: true)
        return ContextWindow.messages(from: page.items, excluding: messageID)
    }

    func deleteConversation(_ id: UUID) throws {
        try opened().deleteConversation(id)
        // Database deletion commits first. A failed file deletion is retried later.
        try clearTemporaryAudio(in: id)
        try pruneLegacyOrphans()
    }

    func eraseHistory() throws {
        database = nil
        for suffix in ["", "-wal", "-shm"] {
            let url = URL(fileURLWithPath: repository.databaseURL.path + suffix)
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        }
        if FileManager.default.fileExists(atPath: repository.legacyHistoryURL.path) {
            try FileManager.default.removeItem(at: repository.legacyHistoryURL)
        }
        for directory in [repository.audioDirectory, repository.temporaryAudioDirectory] {
            if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
        }
        try repository.prepare()
        let opened = try ConversationDatabase(url: repository.databaseURL)
        try opened.migrateLegacy(from: repository.legacyHistoryURL)
        database = opened
    }

    func recordingURL(in id: UUID) throws -> URL {
        try Task.checkCancellation()
        try ensureAudioBudget(adding: ResourcePolicy.maximumAudioBytes)
        let directory = sessionDirectory.appending(path: id.uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appending(path: "temp-input-\(UUID().uuidString).m4a")
    }

    func readRecording(_ url: URL) throws -> Data {
        try Task.checkCancellation()
        guard isInsideSession(url), let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size > 0, size <= ResourcePolicy.maximumAudioBytes else { throw ArchiveError.invalidAudio }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count <= ResourcePolicy.maximumAudioBytes else { throw ArchiveError.invalidAudio }
        try Task.checkCancellation()
        return data
    }

    func encodedRecording(_ url: URL) throws -> String {
        try readRecording(url).base64EncodedString()
    }

    func saveReply(_ encoded: String, mimeType: String?, in id: UUID) throws -> String {
        try Task.checkCancellation()
        guard encoded.utf8.count <= (ResourcePolicy.maximumAudioBytes + 2) / 3 * 4,
              let data = Data(base64Encoded: encoded), !data.isEmpty,
              data.count <= ResourcePolicy.maximumAudioBytes else { throw ArchiveError.invalidAudio }
        try ensureAudioBudget(adding: data.count)
        let ext = mimeType == "audio/wav" ? "wav" : mimeType == "audio/mp4" ? "m4a" : "mp3"
        let directory = sessionDirectory.appending(path: id.uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = "temp-reply-\(UUID().uuidString).\(ext)"
        let url = directory.appending(path: filename)
        var options: Data.WritingOptions = [.atomic]
        #if os(iOS)
        options.insert(.completeFileProtection)
        #endif
        try data.write(to: url, options: options)
        do { try Task.checkCancellation() }
        catch { try? FileManager.default.removeItem(at: url); throw error }
        return filename
    }

    func playbackURL(_ filename: String, in id: UUID) throws -> URL {
        let name = URL(fileURLWithPath: filename).lastPathComponent
        let url = name.hasPrefix("temp-")
            ? sessionDirectory.appending(path: id.uuidString).appending(path: name)
            : repository.audioURL(name)
        let root = name.hasPrefix("temp-") ? sessionDirectory : repository.audioDirectory
        guard url.resolvingSymlinksInPath().path.hasPrefix(root.resolvingSymlinksInPath().path + "/"),
              FileManager.default.fileExists(atPath: url.path),
              let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size > 0, size <= ResourcePolicy.maximumAudioBytes else { throw ArchiveError.invalidAudio }
        return url
    }

    @discardableResult func removeTemporaryAudio(_ filename: String, in id: UUID) throws -> Bool {
        guard filename.hasPrefix("temp-") else { return false }
        let url = sessionDirectory.appending(path: id.uuidString).appending(path: URL(fileURLWithPath: filename).lastPathComponent)
        if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        return true
    }

    func clearTemporaryAudio(in id: UUID? = nil) throws {
        let directory = id.map { sessionDirectory.appending(path: $0.uuidString) } ?? sessionDirectory
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
    }

    func removeExpiredAudio(protecting names: Set<String>, now: Date = .now) throws -> Set<String> {
        var removed: Set<String> = []
        try enumerateFiles(in: sessionDirectory) { url, values in
            guard !names.contains(url.lastPathComponent),
                  let date = values.contentModificationDate,
                  now.timeIntervalSince(date) >= ResourcePolicy.temporaryAudioLifetime else { return }
            try FileManager.default.removeItem(at: url)
            removed.insert(url.lastPathComponent)
        }
        return removed
    }

    func usage() throws -> StorageUsage {
        var usage = StorageUsage()
        for suffix in ["", "-wal", "-shm"] {
            let url = URL(fileURLWithPath: repository.databaseURL.path + suffix)
            usage.textBytes += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        try enumerateFiles(in: repository.audioDirectory) { _, values in usage.legacyAudioBytes += Int64(values.fileSize ?? 0) }
        try enumerateFiles(in: repository.temporaryAudioDirectory) { _, values in usage.temporaryAudioBytes += Int64(values.fileSize ?? 0) }
        return usage
    }

    private func ensureAudioBudget(adding bytes: Int) throws {
        let available = try? repository.directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
        if let available, available < 30_000_000 { throw ArchiveError.lowDisk }
        var used: Int64 = 0
        var count = 0
        try enumerateFiles(in: sessionDirectory) { _, values in used += Int64(values.fileSize ?? 0); count += 1 }
        guard count < ResourcePolicy.maximumTemporaryAudioFiles,
              used + Int64(bytes) <= Int64(ResourcePolicy.maximumTemporaryAudioBytes) else { throw ArchiveError.audioLimit }
    }

    private func isInsideSession(_ url: URL) -> Bool {
        url.resolvingSymlinksInPath().path.hasPrefix(sessionDirectory.resolvingSymlinksInPath().path + "/")
    }

    private func clearOldTemporarySessions() throws {
        guard let enumerator = FileManager.default.enumerator(at: repository.temporaryAudioDirectory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsSubdirectoryDescendants]) else { return }
        for case let url as URL in enumerator where url.lastPathComponent != sessionID && url.lastPathComponent.hasPrefix("session-") {
            try FileManager.default.removeItem(at: url)
        }
    }

    private func pruneLegacyOrphans() throws {
        let database = try opened()
        try enumerateFiles(in: repository.audioDirectory) { url, _ in
            guard ["m4a", "mp3", "wav"].contains(url.pathExtension.lowercased()),
                  !(try database.referencesAudio(url.lastPathComponent)) else { return }
            try FileManager.default.removeItem(at: url)
        }
    }

    private func enumerateFiles(in directory: URL, _ work: (URL, URLResourceValues) throws -> Void) throws {
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey]) else { return }
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else { continue }
            try work(url, values)
        }
    }
}
