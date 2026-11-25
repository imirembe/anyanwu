import Foundation

enum AppTab: String, CaseIterable, Sendable {
    case speak = "Speak", conversations = "Conversations", settings = "Settings"
    var symbol: String {
        switch self {
        case .speak: "waveform"
        case .conversations: "bubble.left.and.bubble.right"
        case .settings: "slider.horizontal.3"
        }
    }
}

enum ServiceMode: String, Codable, CaseIterable, Sendable {
    case apple = "Apple", preview = "Preview", connected = "Connected"
    func title(in language: InterfaceLanguage) -> String {
        switch self {
        case .apple: language.text("On iPhone", "Na iPhone")
        case .preview: language.text("Preview", "Nlele")
        case .connected: language.text("Service", "Ọrụ")
        }
    }
}

enum InterfaceLanguage: String, Codable, CaseIterable, Sendable {
    case igbo = "ig", english = "en"
    var title: String { self == .igbo ? "Igbo" : "English" }
    var locale: Locale { Locale(identifier: self == .igbo ? "ig_NG" : "en_GB") }
    var greeting: String { text("Hello.", "Ndewo.") }
    func text(_ english: String, _ igbo: String) -> String { self == .igbo ? igbo : english }

    func title(for tab: AppTab) -> String {
        switch tab {
        case .speak: text("Home", "Mbido")
        case .conversations: text("Conversations", "Mkparịta ụka")
        case .settings: text("Settings", "Ntọala")
        }
    }
}

struct Message: Identifiable, Codable, Equatable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant }
    let id: UUID
    let role: Role
    var text: String
    let createdAt: Date
    var isPreview: Bool
    var audioFilename: String?
    var isVoicePlaceholder: Bool { audioFilename != nil && text.hasPrefix("Voice message ·") }

    func displayText(in language: InterfaceLanguage) -> String {
        isVoicePlaceholder ? text.replacingOccurrences(of: "Voice message", with: language.text("Voice message", "Ozi olu")) : text
    }

    init(id: UUID = UUID(), role: Role, text: String, createdAt: Date = .now,
         isPreview: Bool = false, audioFilename: String? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.isPreview = isPreview
        self.audioFilename = audioFilename
    }
}

struct Conversation: Identifiable, Codable, Equatable, Sendable {
    var id = UUID()
    var createdAt = Date()
    var messages: [Message] = []
    var title: String {
        let text = messages.first(where: { $0.role == .user })?.text ?? "New conversation"
        return String(text.prefix(65))
    }
    var updatedAt: Date { messages.last?.createdAt ?? createdAt }
    func title(in language: InterfaceLanguage) -> String {
        guard let message = messages.first(where: { $0.role == .user }) else {
            return language.text("New conversation", "Mkparịta ụka ọhụrụ")
        }
        return String(message.displayText(in: language).prefix(65))
    }
}

struct AssistantSettings: Codable, Equatable, Sendable {
    var mode: ServiceMode = .apple
    var appleModeVersion = 1
    var automaticallyPrepareSpeech = true
    var serverURL = ""
    var interfaceLanguage: InterfaceLanguage = .igbo
    var autoplayReplies = true
    var saveHistory = true

    init() {}

    private enum CodingKeys: String, CodingKey {
        case mode, appleModeVersion, automaticallyPrepareSpeech, serverURL, interfaceLanguage, autoplayReplies, saveHistory
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let previousMode = try values.decodeIfPresent(ServiceMode.self, forKey: .mode) ?? .apple
        let version = try values.decodeIfPresent(Int.self, forKey: .appleModeVersion) ?? 0
        // Move the old, unconfigured preview to the requested Apple route.
        // Preserve existing service connections and future explicit previews.
        mode = version == 0 && previousMode == .preview ? .apple : previousMode
        appleModeVersion = 1
        automaticallyPrepareSpeech = try values.decodeIfPresent(Bool.self, forKey: .automaticallyPrepareSpeech) ?? true
        serverURL = try values.decodeIfPresent(String.self, forKey: .serverURL) ?? ""
        interfaceLanguage = try values.decodeIfPresent(InterfaceLanguage.self, forKey: .interfaceLanguage) ?? .igbo
        autoplayReplies = try values.decodeIfPresent(Bool.self, forKey: .autoplayReplies) ?? true
        saveHistory = try values.decodeIfPresent(Bool.self, forKey: .saveHistory) ?? true
    }
}

enum ResourcePolicy {
    static let historyPageSize = 30
    static let messagePageSize = 50
    static let maximumSessionMessages = 200
    static let contextMessageCount = 12
    static let contextCharacterCount = 24_000
    static let maximumAudioBytes = 6_000_000
    static let maximumResponseBytes = 9_000_000
    static let maximumTemporaryAudioBytes = 24_000_000
    static let maximumTemporaryAudioFiles = 12
    static let temporaryAudioLifetime: TimeInterval = 10 * 60
    static let onDeviceInstallFreeBytes: Int64 = 2_000_000_000
    static let onDeviceDownloadBytes: Int64 = 330_000_000
    static let onDeviceCompiledModelBytes: Int64 = 700_000_000
    static let onDeviceInputSamples = 160_000
    static let onDeviceVocabularyCount = 10_288
    static let onDeviceMaximumFrames = 512
}

struct ConversationSummary: Identifiable, Equatable, Sendable {
    let id: UUID
    let createdAt: Date
    let updatedAt: Date
    let firstUserText: String
    let messageCount: Int

    func title(in language: InterfaceLanguage) -> String {
        firstUserText.isEmpty ? language.text("New conversation", "Mkparịta ụka ọhụrụ") : firstUserText
    }
}

struct ArchivePage<Value: Sendable>: Sendable {
    let items: [Value]
    let hasOlder: Bool
    let hasNewer: Bool
}

struct StorageUsage: Sendable {
    var textBytes: Int64 = 0
    var legacyAudioBytes: Int64 = 0
    var temporaryAudioBytes: Int64 = 0
    var totalBytes: Int64 { textBytes + legacyAudioBytes + temporaryAudioBytes }
}

enum ContextWindow {
    static func messages(from source: [Message], excluding id: UUID? = nil) -> [TurnRequest.ContextMessage] {
        var result: [TurnRequest.ContextMessage] = []
        var remaining = ResourcePolicy.contextCharacterCount
        for message in source.reversed() where message.id != id && !message.isPreview && !message.isVoicePlaceholder {
            guard remaining > 0, result.count < ResourcePolicy.contextMessageCount else { break }
            let content = String(message.text.prefix(min(8_000, remaining)))
            remaining -= content.count
            result.append(.init(role: message.role.rawValue, content: content))
        }
        return Array(result.reversed())
    }
}

enum PreviewReplies {
    static func reply(to text: String) -> String {
        let normalized = text.lowercased()
        if normalized.contains("thank") || normalized.contains("daal") || normalized.contains("dalụ") {
            return "Daalụ!"
        }
        if normalized.contains("greet") || normalized.contains("hello") || normalized.contains("ndewo") {
            return "Ndewo! Kedu ka ị mere?"
        }
        return "Ndewo! Abụ m Anyanwụ. Nke a bụ nzaghachi nlele."
    }
}
