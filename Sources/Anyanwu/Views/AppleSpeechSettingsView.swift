import SwiftUI

struct AppleSpeechSettingsView: View {
    @Environment(AssistantStore.self) private var assistant
    @State private var confirmRemove = false
    private var apple: AppleAssistant { assistant.appleAssistant }
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(language.text("Your voice is transcribed on this iPhone. Igbo, English and mixed speech use the same recognition model.", "A na-ede olu gị na iPhone a. Igbo, Bekee na okwu agwakọtara na-eji otu ụdị ịghọta olu."))
                .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
            Toggle(language.text("Prepare speech automatically", "Kwadebe olu na-akpaghị aka"), isOn: Binding(
                get: { assistant.settings.automaticallyPrepareSpeech },
                set: { enabled in
                    if enabled {
                        var value = assistant.settings; value.automaticallyPrepareSpeech = true
                        assistant.saveSettings(value)
                    } else { assistant.cancelSpeechPreparation() }
                }
            )).font(AnyanwuTypography.caption)
            status(language.text("Speech recognition", "Ịghọta olu"), value: apple.capabilities.speechInstalled
                   ? language.text("Installed", "Ebudatala") : language.text("Download needed", "A ga-ebudata ya"), ready: apple.capabilities.speechInstalled)
            if apple.isInstalling {
                ProgressView(value: apple.installProgress)
                    .tint(AnyanwuTheme.forest).accessibilityLabel(language.text("Speech download progress", "Ọganihu ibudata olu"))
                Text(apple.installProgress >= 0.9 ? language.text("Preparing on this iPhone…", "A na-akwado ya na iPhone a…")
                     : language.text("Downloading… \(Int(apple.installProgress / 0.9 * 100))%", "A na-ebudata… \(Int(apple.installProgress / 0.9 * 100))%"))
                    .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                Button(language.text("Cancel download", "Kagbuo ibudata")) { assistant.cancelSpeechPreparation() }
                    .font(AnyanwuTypography.caption)
            } else if !apple.capabilities.speechInstalled {
                if apple.capabilities.isSimulator {
                    Text(language.text("Speech downloads are enabled on a real iPhone. Large models stay off your Mac.", "A na-ebudata ọrụ olu na iPhone n'ezie. A naghị etinye nnukwu ụdị na Mac gị."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                } else {
                    Button { assistant.prepareSpeech() } label: {
                        Label(language.text("Download speech · 327 MB", "Budata olu · 327 MB"), systemImage: "arrow.down.circle")
                    }.accessibilityIdentifier("download-apple-speech")
                    Text(language.text("Speech prepares automatically on Wi-Fi and stays saved for later. Keep Anyanwụ open during setup and allow 2 GB free.", "A na-akwado olu na-akpaghị aka na Wi-Fi, a na-echekwa ya maka oge ọzọ. Hapụ Anyanwụ ka ọ ghe oghe n'oge nhazi, ma mee ka ohere 2 GB dị."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                }
            } else {
                HStack {
                    Text(ByteCountFormatter.string(fromByteCount: apple.capabilities.modelBytes, countStyle: .file))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                    Spacer()
                    Button(language.text("Remove download", "Hichapụ ihe e budatara"), role: .destructive) { confirmRemove = true }
                        .font(AnyanwuTypography.caption).accessibilityIdentifier("remove-apple-speech")
                }
            }
            Divider()
            status(language.text("Igbo replies", "Nzaghachi Igbo"), value: apple.capabilities.igboGenerationAvailable
                   ? language.text("Apple Intelligence", "Apple Intelligence") : language.text("Prepared examples", "Ihe atụ akwadoro"), ready: apple.capabilities.igboGenerationAvailable)
            if !apple.capabilities.igboGenerationAvailable {
                Text(generationDescription).font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
            }
            status(language.text("Igbo speaking voice", "Olu ịgụ Igbo"), value: apple.capabilities.igboVoiceAvailable
                   ? language.text("Available", "Ọ dị") : language.text("Unavailable", "Ọ dịghị"), ready: apple.capabilities.igboVoiceAvailable)
            if !apple.capabilities.igboVoiceAvailable {
                Text(language.text("An Igbo voice model is still needed for spoken replies. Replies stay in Igbo as text.", "A ka chọrọ ụdị olu Igbo maka ịkpọ nzaghachi. Nzaghachi na-adị n'Igbo dịka ederede."))
                    .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
            }
            Button { Task { await apple.refresh() } } label: {
                Label(language.text("Check this iPhone", "Lelee iPhone a"), systemImage: "arrow.clockwise")
            }.font(AnyanwuTypography.caption).disabled(apple.isInstalling)
                .accessibilityIdentifier("check-apple-capabilities")
            if let error = apple.setupError {
                NoticeView(text: error, isError: true) { apple.setupError = nil }
            }
            Link(language.text("About the speech model", "Banyere ụdị olu a"),
                 destination: URL(string: "https://huggingface.co/\(AppleSpeechAssets.repository)/blob/\(AppleSpeechAssets.revision)/README.md")!)
                .font(AnyanwuTypography.caption2)
            Text(language.text("Recognition quality needs testing with real Igbo speakers. This version shows the transcript after recording stops.", "A ga-eji ndị na-asụ Igbo nwalee ogo ịghọta olu. Ụdị a na-egosi ederede mgbe ndekọ kwụsịrị."))
                .font(AnyanwuTypography.caption2).foregroundStyle(AnyanwuTheme.muted)
        }
        .task { await apple.refresh() }
        .confirmationDialog(language.text("Remove downloaded speech recognition?", "Hichapụ ọrụ ịghọta olu e budatara?"),
                            isPresented: $confirmRemove, titleVisibility: .visible) {
            Button(language.text("Remove download", "Hichapụ ihe e budatara"), role: .destructive) { assistant.removeOnDeviceSpeech() }
        } message: {
            Text(language.text("Your conversations stay saved. You can download speech again later.", "A ga-edobe mkparịta ụka gị. Ị nwere ike ibudata olu ọzọ ma emechaa."))
        }
    }

    private func status(_ title: String, value: String, ready: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(AnyanwuTypography.sans(14, weight: .medium, relativeTo: .subheadline))
            Label(value, systemImage: ready ? "checkmark.circle" : "circle.dashed")
                .font(AnyanwuTypography.caption).foregroundStyle(ready ? AnyanwuTheme.forest : AnyanwuTheme.muted)
        }
    }

    private var generationDescription: String {
        switch apple.capabilities.generationStatus {
        case "igboUnsupported": language.text("Apple’s on-device reply model does not list Igbo support on this device.", "Ụdị nzaghachi Apple na ngwaọrụ a anaghị akwado Igbo.")
        case "disabled": language.text("Apple Intelligence is turned off. Igbo support is checked separately when it is available.", "Agbanyụọla Apple Intelligence. A ga-enyocha nkwado Igbo mgbe ọ dị.")
        case "notReady": language.text("Apple’s on-device model is not ready. No model download is requested by this check.", "Ụdị Apple dị na ngwaọrụ adịghị njikere. Nlele a anaghị ebudata ụdị ọ bụla.")
        default: language.text("Apple’s on-device Igbo reply generation is unavailable here. Prepared replies remain labelled Preview.", "Enweghị ike iji Apple mepụta nzaghachi Igbo ebe a. A na-akanye nzaghachi akwadoro Nlele.")
        }
    }
}
