import SwiftUI

struct RootView: View {
    @Environment(AssistantStore.self) private var assistant
    @Environment(\.scenePhase) private var scenePhase
    @State private var launch = VoiceLaunchRequest.shared
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    private var isHome: Bool { assistant.tab == .speak && !assistant.showConversation }

    var body: some View {
        VStack(spacing: 0) {
            header
            Group {
                switch assistant.tab {
                case .speak:
                    if assistant.showConversation { ConversationView() }
                    else { SpeakView() }
                case .conversations: HistoryView()
                case .settings: SettingsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(AnyanwuTheme.paper.ignoresSafeArea())
        .tint(AnyanwuTheme.forest)
        .font(AnyanwuTypography.body)
        .environment(\.locale, language.locale)
        .onChange(of: launch.pending) { _, _ in handleVoiceLaunch() }
        .onChange(of: assistant.isPreparingStorage) { _, preparing in if !preparing { handleVoiceLaunch() } }
        .onChange(of: scenePhase) { _, phase in if phase == .active { handleVoiceLaunch() } }
        .task { handleVoiceLaunch() }
        .alert(assistant.error == nil ? language.text("Recording", "Ndekọ")
               : language.text("Something went wrong", "Enwere nsogbu"), isPresented: Binding(
            get: { isHome && (assistant.error != nil || assistant.recordingNotice != nil) },
            set: { if !$0 { assistant.error = nil; assistant.recordingNotice = nil } }
        )) {
            Button(language.text("OK", "Ọ dị mma")) { assistant.error = nil; assistant.recordingNotice = nil }
        } message: { Text(assistant.error ?? assistant.recordingNotice ?? "") }
    }

    private var header: some View {
        HStack(spacing: 4) {
            Menu {
                ForEach(InterfaceLanguage.allCases, id: \.self) { option in
                    Button {
                        var settings = assistant.settings
                        settings.interfaceLanguage = option
                        assistant.saveSettings(settings)
                    } label: {
                        if language == option { Label(option.title, systemImage: "checkmark") }
                        else { Text(option.title) }
                    }
                    .accessibilityIdentifier("language-\(option.rawValue)")
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "globe").font(.system(size: 14, weight: .regular))
                    Text(language.title).font(AnyanwuTypography.sans(14, weight: .medium, relativeTo: .subheadline))
                        .lineLimit(1).minimumScaleFactor(0.7)
                    Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold))
                }
                .padding(.horizontal, 14).frame(minHeight: 44)
                .background(.white.opacity(0.55), in: Capsule())
                .overlay(Capsule().stroke(AnyanwuTheme.line, lineWidth: 0.8))
            }
            .accessibilityLabel(language.text("App language", "Asụsụ ngwa"))
            .accessibilityValue(language.title)
            .accessibilityIdentifier("app-language")

            Spacer()
            if !isHome { navigationButton(.speak, symbol: "house") }
            navigationButton(.conversations, symbol: "bubble.left.and.bubble.right")
            navigationButton(.settings, symbol: "slider.horizontal.3")
        }
        .foregroundStyle(AnyanwuTheme.forest)
        .padding(.horizontal, 24).padding(.top, 10).padding(.bottom, 6)
    }

    private func navigationButton(_ tab: AppTab, symbol: String) -> some View {
        Button {
            assistant.navigate(to: tab)
        } label: {
            Image(systemName: symbol).font(.system(size: 18, weight: .regular))
                .frame(width: 44, height: 44)
                .background(assistant.tab == tab ? AnyanwuTheme.pale : .clear, in: Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(language.title(for: tab))
        .accessibilityIdentifier("tab-\(tab.rawValue.lowercased())")
    }

    private func handleVoiceLaunch() {
        guard launch.pending, scenePhase == .active, !assistant.isPreparingStorage else { return }
        launch.pending = false
        assistant.newConversation()
        assistant.toggleRecording()
    }
}
