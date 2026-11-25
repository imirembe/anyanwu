import SwiftUI

/// Setup belongs to the app, so progress survives moving between screens.
struct SpeechPreparationView: View {
    @Environment(AssistantStore.self) private var assistant
    private var apple: AppleAssistant { assistant.appleAssistant }
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(language.text("Preparing speech", "A na-akwado olu"), systemImage: "arrow.down.circle")
                .font(AnyanwuTypography.sans(15, weight: .medium, relativeTo: .subheadline))
            if apple.isInstalling {
                ProgressView(value: apple.installProgress).tint(AnyanwuTheme.forest)
                    .accessibilityLabel(language.text("Speech setup progress", "Ọganihu nhazi olu"))
                Text(apple.installProgress >= 0.9
                     ? language.text("Almost ready…", "Ọ fọrọ nke nta ka ọ dị njikere…")
                     : language.text("Downloading \(Int(apple.installProgress / 0.9 * 100))% · 327 MB", "A na-ebudata \(Int(apple.installProgress / 0.9 * 100))% · 327 MB"))
                Button(language.text("Cancel", "Kagbuo")) { assistant.cancelSpeechPreparation() }
            } else if apple.capabilities.isSimulator {
                Text(language.text("Speech setup is available on your iPhone.", "Nhazi olu dị na iPhone gị."))
            } else {
                if let error = apple.setupError { Text(error) }
                else {
                    Text(apple.isWaitingForWiFi
                         ? language.text("Connect to Wi-Fi. Speech will download automatically, then work offline.", "Jikọọ na Wi-Fi. A ga-ebudata olu na-akpaghị aka, ọ ga-arụkwa ọrụ na-enweghị ịntanetị.")
                         : language.text("Download speech once to use your microphone. It stays saved on this iPhone.", "Budata olu otu ugboro iji igwe okwu gị. A ga-echekwa ya na iPhone a."))
                }
                Button(language.text("Prepare speech · 327 MB", "Kwadebe olu · 327 MB")) { assistant.prepareSpeech() }
                    .accessibilityIdentifier("prepare-speech")
            }
        }
        .font(AnyanwuTypography.caption)
        .foregroundStyle(AnyanwuTheme.forest)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(AnyanwuTheme.pale, in: RoundedRectangle(cornerRadius: 20))
        .accessibilityIdentifier("speech-preparation")
    }
}
