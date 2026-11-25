import AppIntents
import Observation

@MainActor @Observable
final class VoiceLaunchRequest {
    static let shared = VoiceLaunchRequest()
    var pending = false
}

struct TalkToAnyanwuIntent: AppIntent {
    static var title: LocalizedStringResource = "Talk to Anyanwụ"
    static var description = IntentDescription("Open Anyanwụ to start an Igbo voice conversation.")
    static var openAppWhenRun = true

    @MainActor func perform() async throws -> some IntentResult {
        VoiceLaunchRequest.shared.pending = true
        return .result()
    }
}

struct AnyanwuShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: TalkToAnyanwuIntent(),
            phrases: ["Talk to \(.applicationName)", "Start a conversation with \(.applicationName)"],
            shortTitle: "Talk to Anyanwụ",
            systemImageName: "waveform"
        )
    }
}
