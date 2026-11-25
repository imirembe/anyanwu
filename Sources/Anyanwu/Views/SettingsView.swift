import SwiftUI

struct SettingsView: View {
    @Environment(AssistantStore.self) private var assistant
    @State private var preferences = AssistantSettings()
    @State private var token = ""
    @State private var connectionStatus: String?
    @State private var checking = false
    @State private var confirmDelete = false
    @State private var confirmDisableHistory = false
    @State private var settingsError: String?
    @State private var saving = false
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text(language.text("Settings", "Ntọala"))
                    .font(AnyanwuTypography.serif(32, relativeTo: .largeTitle)).foregroundStyle(AnyanwuTheme.forest).padding(.top, 16)
                section(language.text("VOICE", "OLU")) {
                    Text(language.text("Speak in Igbo or English. Anyanwụ always replies in Igbo.", "Kwuo okwu n'Igbo ma ọ bụ Bekee. Anyanwụ na-aza n'Igbo mgbe niile."))
                    Divider()
                    Toggle(language.text("Play spoken replies", "Kpọọ nzaghachi olu"), isOn: $preferences.autoplayReplies)
                }
                section(language.text("CONNECTION", "NJIKỌ")) {
                    Picker(language.text("Service mode", "Ụdị ọrụ"), selection: $preferences.mode) {
                        ForEach(ServiceMode.allCases, id: \.self) { mode in Text(mode.title(in: language)).tag(mode) }
                    }.pickerStyle(.segmented).accessibilityIdentifier("mode-picker")
                    Text(preferences.mode == .preview
                         ? language.text("Preview uses prepared replies.", "Nlele na-eji nzaghachi akwadoro.")
                         : preferences.mode == .apple
                         ? language.text("Process speech on this iPhone. No voice-service account is needed.", "Hazie olu na iPhone a. A chọghị akaụntụ ọrụ olu.")
                         : language.text("Your service processes messages and recordings.", "Ọrụ gị na-ahazi ozi na ndekọ olu."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                    if preferences.mode == .connected {
                        Divider()
                        Text(language.text("Service address", "Adreesị ọrụ")).font(AnyanwuTypography.sans(12, weight: .medium, relativeTo: .caption))
                        TextField("https://your-anyanwu-service.com", text: $preferences.serverURL)
                            .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                            .font(AnyanwuTypography.sans(13, relativeTo: .footnote)).accessibilityIdentifier("server-url")
                        Text(language.text("Simulator: http://localhost:8787. Use HTTPS on a real iPhone.", "Simulator: http://localhost:8787. Jiri HTTPS na iPhone n'ezie."))
                            .font(AnyanwuTypography.caption2).foregroundStyle(AnyanwuTheme.muted)
                        Divider()
                        SecureField(language.text("Access token", "Igodo njikọ"), text: $token)
                            .textInputAutocapitalization(.never).autocorrectionDisabled().font(AnyanwuTypography.subheadline)
                            .accessibilityIdentifier("access-token")
                        Button { checkConnection() } label: {
                            HStack(spacing: 8) {
                                if checking { ProgressView().controlSize(.small) }
                                Label(checking ? language.text("Checking…", "A na-enyocha…")
                                      : language.text("Check connection", "Lelee njikọ"), systemImage: "network")
                            }
                        }.font(AnyanwuTypography.sans(13, weight: .medium, relativeTo: .footnote)).disabled(checking)
                            .accessibilityIdentifier("check-connection")
                    }
                    if preferences.mode == .apple {
                        Divider()
                        AppleSpeechSettingsView()
                    }
                    if let connectionStatus {
                        Text(connectionStatus).font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.forest)
                            .accessibilityIdentifier("connection-status")
                    }
                }
                section(language.text("HISTORY", "AKỤKỌ")) {
                    Toggle(language.text("Save conversations", "Chekwaa mkparịta ụka"), isOn: Binding(
                        get: { preferences.saveHistory }, set: { value in
                            if !value { confirmDisableHistory = true } else { preferences.saveHistory = true }
                        }
                    ))
                    Text(language.text("Conversation text stays on this iPhone. New recordings are temporary. Connected services have their own data policies.", "Ederede mkparịta ụka na-adị na iPhone a. Ndekọ ọhụrụ bụ nke nwa oge. Ọrụ ejikọtara nwere iwu nke ha maka data."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                    Divider()
                    Button(language.text("Delete history and recordings", "Hichapụ akụkọ na ndekọ"), role: .destructive) { confirmDelete = true }
                        .font(AnyanwuTypography.sans(13, relativeTo: .footnote)).accessibilityIdentifier("clear-history")
                }
                section(language.text("STORAGE", "NCHEKWA")) {
                    storageRow(language.text("Conversation text", "Ederede mkparịta ụka"), bytes: assistant.storageUsage.textBytes)
                    storageRow(language.text("Temporary audio", "Ụda nwa oge"), bytes: assistant.storageUsage.temporaryAudioBytes)
                    if assistant.appleAssistant.capabilities.modelBytes > 0 {
                        storageRow(language.text("Speech recognition model", "Ụdị ịghọta olu"), bytes: assistant.appleAssistant.capabilities.modelBytes)
                    }
                    if assistant.storageUsage.legacyAudioBytes > 0 {
                        storageRow(language.text("Previously saved audio", "Ụda echekwara mbụ"), bytes: assistant.storageUsage.legacyAudioBytes)
                    }
                    Text(language.text("Voice input is removed after transcription. Spoken replies are removed after playback. Unused clips expire after 10 minutes or when you leave the conversation.", "A na-ehichapụ olu ị zitere mgbe e dechara ya. A na-ehichapụ nzaghachi olu mgbe akpọchara ya. Ndekọ a na-ejighị na-apụ mgbe nkeji iri gachara ma ọ bụ mgbe ị hapụrụ mkparịta ụka."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                    Button(language.text("Clear temporary audio", "Hichapụ ụda nwa oge")) { assistant.clearTemporaryAudio() }
                        .font(AnyanwuTypography.sans(13, relativeTo: .footnote))
                }
                section("SIRI") {
                    Text("“Siri, talk to Anyanwụ”").font(AnyanwuTypography.serif(17, relativeTo: .body))
                    Text(language.text("Use this supported Siri phrase to open Anyanwụ. Speak Igbo or English once the app opens.", "Jiri ahịrịokwu Siri a mepee Anyanwụ. Mgbe ngwa meghere, kwuo okwu n'Igbo ma ọ bụ Bekee."))
                        .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                }
                if let settingsError { NoticeView(text: settingsError, isError: true) { self.settingsError = nil } }
                Button { save() } label: {
                    Text(language.text("Save", "Chekwaa")).font(AnyanwuTypography.sans(14, weight: .semibold, relativeTo: .subheadline))
                        .frame(maxWidth: .infinity).padding(.vertical, 17)
                        .background(AnyanwuTheme.forest, in: Capsule()).foregroundStyle(AnyanwuTheme.paper)
                }.disabled(saving || assistant.isPreparingStorage).accessibilityIdentifier("save-settings")
                Text("Anyanwụ · 0.1.0").font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted)
                    .frame(maxWidth: .infinity).padding(.bottom, 24)
            }.padding(.horizontal, 24)
        }
        .scrollIndicators(.hidden)
        .onAppear { preferences = assistant.settings; token = SecureToken.read() ?? "" }
        .task { await assistant.refreshStorageUsage(); await assistant.appleAssistant.refresh() }
        .onChange(of: language) { _, _ in connectionStatus = nil; settingsError = nil }
        .confirmationDialog(language.text("Delete all history and recordings?", "Hichapụ akụkọ na ndekọ niile?"), isPresented: $confirmDelete, titleVisibility: .visible) {
            Button(language.text("Delete", "Hichapụ"), role: .destructive) { assistant.clearHistory() }
            Button(language.text("Cancel", "Kagbuo"), role: .cancel) {}
        }
        .confirmationDialog(language.text("Turn off history and delete saved recordings?", "Gbanyụọ akụkọ ma hichapụ ndekọ echekwara?"), isPresented: $confirmDisableHistory, titleVisibility: .visible) {
            Button(language.text("Turn off and delete", "Gbanyụọ ma hichapụ"), role: .destructive) {
                preferences.saveHistory = false
                var updated = assistant.settings
                updated.saveHistory = false
                assistant.saveSettings(updated)
            }
            Button(language.text("Cancel", "Kagbuo"), role: .cancel) {}
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Eyebrow(title: title)
            VStack(alignment: .leading, spacing: 13, content: content)
                .font(AnyanwuTypography.sans(14, relativeTo: .subheadline)).foregroundStyle(AnyanwuTheme.ink)
                .padding(17).background(.white.opacity(0.63), in: RoundedRectangle(cornerRadius: 20))
                .overlay(RoundedRectangle(cornerRadius: 20).stroke(AnyanwuTheme.line.opacity(0.65), lineWidth: 0.7))
        }
    }

    private func errorText(_ error: Error) -> String {
        (error as? AssistantServiceError)?.message(in: language)
            ?? language.text(error.localizedDescription, "Enweghị ike imecha ihe a. Biko nwaa ọzọ.")
    }

    private func storageRow(_ title: String, bytes: Int64) -> some View {
        HStack { Text(title); Spacer(); Text(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)).foregroundStyle(AnyanwuTheme.muted) }
    }

    private func save() {
        do {
            if preferences.mode == .connected { _ = try AssistantService.validatedURL(preferences.serverURL) }
            guard SecureToken.save(token.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                settingsError = language.text("Couldn’t save the access token securely.", "Enweghị ike ichekwa igodo njikọ n'enweghị nsogbu.")
                return
            }
            // The top-left menu owns this value, even while this form is open.
            preferences.interfaceLanguage = assistant.settings.interfaceLanguage
            saving = true
            assistant.error = nil
            let task = assistant.saveSettings(preferences)
            Task {
                await task.value
                saving = false
                settingsError = assistant.error
                if settingsError == nil { connectionStatus = language.text("Saved", "Echekwara") }
            }
        } catch { settingsError = errorText(error) }
    }

    private func checkConnection() {
        guard SecureToken.save(token.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            settingsError = language.text("Couldn’t store the access token securely.", "Enweghị ike ichekwa igodo njikọ n'enweghị nsogbu.")
            return
        }
        checking = true
        connectionStatus = nil
        Task {
            defer { checking = false }
            do { connectionStatus = try await AssistantService().check(baseURL: preferences.serverURL, interfaceLanguage: language) }
            catch { settingsError = errorText(error) }
        }
    }
}
