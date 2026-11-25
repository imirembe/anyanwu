import SwiftUI

struct SpeakView: View {
    @Environment(AssistantStore.self) private var assistant
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 36)
            Text(language.greeting)
                .font(AnyanwuTypography.serif(52, relativeTo: .largeTitle))
                .foregroundStyle(AnyanwuTheme.forest)
                .lineLimit(1).minimumScaleFactor(0.7)
                .accessibilityIdentifier("home-greeting")
            Spacer().frame(height: 28)
            Button { assistant.toggleRecording() } label: {
                Image(systemName: "sun.max.fill")
                    .font(.system(size: 84, weight: .regular))
                    .symbolRenderingMode(.monochrome)
                    .foregroundStyle(Color(hex: 0xF0BC36))
                    .shadow(color: AnyanwuTheme.gold.opacity(0.16), radius: 12, y: 3)
                    .frame(width: 112, height: 112)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(assistant.isPreparingStorage || assistant.isStartingRecording || assistant.isWorking)
            .accessibilityIdentifier("record-button")
            .accessibilityLabel(language.text("Start speaking", "Malite ikwu okwu"))
            .accessibilityHint(language.text("Start a voice conversation in Igbo or English.", "Malite mkparịta ụka olu n'Igbo ma ọ bụ Bekee."))
            Spacer(minLength: 36)
            Button { assistant.openComposer() } label: {
                Image(systemName: "keyboard").font(.system(size: 19, weight: .regular))
                    .foregroundStyle(AnyanwuTheme.muted).frame(width: 44, height: 44)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(language.text("Type a message", "Dee ozi"))
            .accessibilityIdentifier("open-composer")
            .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ComposerView: View {
    @Environment(AssistantStore.self) private var assistant
    @FocusState private var focused: Bool
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    var body: some View {
        @Bindable var assistant = assistant
        HStack(spacing: 10) {
            TextField(language.text("Type a message…", "Dee ozi…"), text: $assistant.draft, axis: .vertical)
                .font(AnyanwuTypography.subheadline).lineLimit(1...4).focused($focused)
                .accessibilityIdentifier("message-input").submitLabel(.send)
                .onSubmit { assistant.sendText(); focused = false }
                .disabled(assistant.recorder.isRecording)
            Button { assistant.sendText(); focused = false } label: {
                Image(systemName: "arrow.up").font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(AnyanwuTheme.paper).frame(width: 32, height: 32)
                    .background(AnyanwuTheme.forest.opacity(canSend ? 1 : 0.25), in: Circle())
            }
            .disabled(!canSend).accessibilityLabel(language.text("Send message", "Ziga ozi"))
            .accessibilityIdentifier("send-message")
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(.white.opacity(0.7), in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(AnyanwuTheme.line, lineWidth: 0.8))
        .task { focused = true }
    }
    private var canSend: Bool {
        !assistant.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !assistant.isWorking && !assistant.recorder.isRecording
            && !assistant.isPreparingStorage && !assistant.isLoadingConversation
    }
}
