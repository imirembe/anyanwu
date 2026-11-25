import Foundation

enum ArchiveError: LocalizedError {
    case database(String), invalidLegacyHistory, unavailable, audioLimit, invalidAudio, lowDisk
    var errorDescription: String? {
        switch self {
        case .database(let detail): "Local history is unavailable: \(detail)"
        case .invalidLegacyHistory: "Existing history could not be migrated safely. The original file has been preserved."
        case .unavailable: "Local history is unavailable. Existing data has been preserved."
        case .audioLimit: "Temporary audio storage is full. Start a new conversation or clear temporary audio in Settings."
        case .invalidAudio: "The recording is missing or exceeds the audio limit."
        case .lowDisk: "There is not enough free space to record. Free some device storage and try again."
        }
    }
}

struct LocalRepository: Sendable {
    let directory: URL
    init(directory: URL? = nil) {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        // Stable path keeps data from installations originally named Ndichie.
        self.directory = directory ?? base.appending(path: "Ndichie", directoryHint: .isDirectory)
    }
    var audioDirectory: URL { directory.appending(path: "Audio", directoryHint: .isDirectory) }
    var temporaryAudioDirectory: URL { directory.appending(path: "TemporaryAudio", directoryHint: .isDirectory) }
    var databaseURL: URL { directory.appending(path: "history.sqlite3") }
    var legacyHistoryURL: URL { directory.appending(path: "conversations.json") }

    func prepare() throws {
        for url in [directory, audioDirectory, temporaryAudioDirectory] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            #if os(iOS)
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: url.path)
            #endif
        }
        var url = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }

    func loadSettings() throws -> AssistantSettings {
        let url = directory.appending(path: "settings.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return .init() }
        guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= 65_536 else { throw ArchiveError.unavailable }
        return try JSONDecoder().decode(AssistantSettings.self, from: Data(contentsOf: url))
    }

    func save<T: Encodable>(_ value: T, file: String) throws {
        try prepare()
        let data = try JSONEncoder().encode(value)
        var options: Data.WritingOptions = [.atomic]
        #if os(iOS)
        options.insert(.completeFileProtection)
        #endif
        try data.write(to: directory.appending(path: URL(fileURLWithPath: file).lastPathComponent), options: options)
    }

    func audioURL(_ filename: String) -> URL { audioDirectory.appending(path: URL(fileURLWithPath: filename).lastPathComponent) }
}
