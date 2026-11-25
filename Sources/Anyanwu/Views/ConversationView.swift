import SwiftUI

struct ConversationView: View {
    @Environment(AssistantStore.self) private var assistant
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    private var recording: Bool { assistant.recorder.isRecording || assistant.isStartingRecording }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(language.text("Conversation", "Mkparịta ụka"))
                    .font(AnyanwuTypography.serif(24, relativeTo: .title2)).foregroundStyle(AnyanwuTheme.forest)
                    .lineLimit(1).minimumScaleFactor(0.6)
                Spacer()
                Button { assistant.newConversation() } label: {
                    Image(systemName: "plus").font(.system(size: 18, weight: .regular)).frame(width: 44, height: 44)
                }
                .accessibilityLabel(language.text("New conversation", "Mkparịta ụka ọhụrụ"))
                .accessibilityIdentifier("new-conversation")
            }.padding(.horizontal, 24).padding(.top, 4).padding(.bottom, 8)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if assistant.isLoadingConversation { ProgressView().frame(maxWidth: .infinity) }
                        if assistant.hasOlderMessages {
                            Button(language.text("Earlier messages", "Ozi gara aga")) { assistant.showOlderMessages() }
                                .font(AnyanwuTypography.caption).frame(maxWidth: .infinity)
                                .disabled(assistant.isLoadingConversation || assistant.isWorking || recording)
                        }
                        ForEach(assistant.messages) { message in MessageBubble(message: message).id(message.id) }
                        if recording { RecordingTranscriptBubble().id("recording") }
                        if assistant.hasNewerMessages {
                            Button(language.text("Later messages", "Ozi sochirinụ")) { assistant.showNewerMessages() }
                                .font(AnyanwuTypography.caption).frame(maxWidth: .infinity)
                                .disabled(assistant.isLoadingConversation || assistant.isWorking || recording)
                        }
                        if assistant.isWorking && assistant.transcribingMessageID == nil {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text(language.text("Preparing a reply…", "A na-akwado nzaghachi…"))
                                    .font(AnyanwuTypography.sans(13, relativeTo: .footnote)).foregroundStyle(AnyanwuTheme.muted)
                                Spacer(minLength: 52)
                            }.padding(.vertical, 4)
                        }
                        if let notice = assistant.recordingNotice { NoticeView(text: notice) }
                        if let error = assistant.error {
                            NoticeView(text: error, isError: true) { assistant.error = nil }
                        }
                        if assistant.settings.mode == .apple && (assistant.appleAssistant.isInstalling || !assistant.appleAssistant.capabilities.speechInstalled) {
                            SpeechPreparationView()
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 24).padding(.top, 16).padding(.bottom, 18)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .scrollIndicators(.hidden).scrollDismissesKeyboard(.interactively)
                .defaultScrollAnchor(.bottom)
                .onChange(of: assistant.messages.last?.id) { _, _ in scrollToBottom(proxy) }
                .onChange(of: assistant.messages.last?.text) { _, _ in scrollToBottom(proxy) }
                .onChange(of: assistant.recordingTranscript) { _, _ in scrollToBottom(proxy) }
                .onChange(of: recording) { _, _ in scrollToBottom(proxy) }
                .onChange(of: assistant.error) { _, _ in scrollToBottom(proxy) }
                .onChange(of: assistant.recordingNotice) { _, _ in scrollToBottom(proxy) }
                .onChange(of: assistant.appleAssistant.isInstalling) { _, _ in scrollToBottom(proxy) }
                .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { VoiceControls() }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
    }
}

struct MessageBubble: View {
    @Environment(AssistantStore.self) private var assistant
    let message: Message
    private var isUser: Bool { message.role == .user }
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    private var transcribing: Bool { message.id == assistant.transcribingMessageID }

    var body: some View {
        HStack(alignment: .bottom, spacing: 0) {
            if isUser { Spacer(minLength: 48) }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
                HStack(spacing: 6) {
                    if !isUser { Image(systemName: "sun.max").foregroundStyle(AnyanwuTheme.gold) }
                    Text(isUser ? language.text("You", "Gị") : "Anyanwụ")
                    Text("·")
                    Text(message.createdAt, style: .time)
                }.font(AnyanwuTypography.sans(10, weight: .medium, relativeTo: .caption2)).foregroundStyle(AnyanwuTheme.muted)

                VStack(alignment: .leading, spacing: 10) {
                    if message.isPreview {
                        Text(language.text("PREVIEW", "NLELE"))
                            .font(AnyanwuTypography.sans(9, weight: .semibold, relativeTo: .caption2)).tracking(0.8)
                            .foregroundStyle(AnyanwuTheme.muted)
                    }
                    if message.isVoicePlaceholder {
                        if transcribing {
                            if !assistant.recordingTranscript.isEmpty {
                                Text(assistant.recordingTranscript).font(AnyanwuTypography.body).lineSpacing(4)
                            }
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text(language.text("Transcribing…", "A na-ede olu…")).font(AnyanwuTypography.sans(13, relativeTo: .footnote))
                            }.foregroundStyle(AnyanwuTheme.muted)
                        } else {
                            Label(message.displayText(in: language), systemImage: "waveform")
                                .font(AnyanwuTypography.sans(14, relativeTo: .subheadline))
                        }
                    } else {
                        Text(message.text).font(AnyanwuTypography.body).lineSpacing(4)
                            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    }
                    if let filename = message.audioFilename, !transcribing {
                        Button { assistant.togglePlayback(filename: filename) } label: {
                            Label(assistant.playback.playingFilename == filename
                                  ? language.text("Stop", "Kwụsị")
                                  : isUser ? language.text("Play recording", "Kpọọ ndekọ") : language.text("Listen", "Gee ntị"),
                                  systemImage: assistant.playback.playingFilename == filename ? "stop.circle.fill" : "play.circle.fill")
                                .font(AnyanwuTypography.sans(12, weight: .medium, relativeTo: .caption))
                        }.accessibilityIdentifier("play-recording")
                        if message.isVoicePlaceholder && assistant.settings.mode != .preview {
                            Button(language.text("Retry transcription", "Nwaa ide olu ọzọ")) { assistant.retryVoice(message) }
                                .font(AnyanwuTypography.caption).disabled(assistant.isWorking)
                        }
                    }
                    if !isUser, message.audioFilename == nil, assistant.settings.mode == .apple,
                       assistant.appleAssistant.capabilities.igboVoiceAvailable {
                        Button { assistant.speakReply(message) } label: {
                            Label(language.text("Listen", "Gee ntị"), systemImage: "speaker.wave.2")
                                .font(AnyanwuTypography.caption)
                        }.disabled(recordingUnavailable)
                    }
                }
                .foregroundStyle(AnyanwuTheme.ink)
                .padding(.horizontal, 17).padding(.vertical, 14)
                .background(isUser ? AnyanwuTheme.pale : .white.opacity(0.8), in: bubbleShape)
                .overlay(bubbleShape.stroke(isUser ? AnyanwuTheme.forest.opacity(0.06) : AnyanwuTheme.line.opacity(0.85), lineWidth: 0.7))
            }
            if !isUser { Spacer(minLength: 48) }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .accessibilityIdentifier("message-\(message.role.rawValue)-\(message.id.uuidString)")
    }

    private var bubbleShape: UnevenRoundedRectangle {
        UnevenRoundedRectangle(topLeadingRadius: 20, bottomLeadingRadius: isUser ? 20 : 6,
                               bottomTrailingRadius: isUser ? 6 : 20, topTrailingRadius: 20)
    }
    private var recordingUnavailable: Bool { assistant.recorder.isRecording || assistant.isStartingRecording || assistant.isWorking }
}

struct RecordingTranscriptBubble: View {
    @Environment(AssistantStore.self) private var assistant
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 48)
            VStack(alignment: .trailing, spacing: 6) {
                Text(language.text("You", "Gị")).font(AnyanwuTypography.sans(10, weight: .medium, relativeTo: .caption2)).foregroundStyle(AnyanwuTheme.muted)
                VStack(alignment: .leading, spacing: 10) {
                    if !assistant.recordingTranscript.isEmpty {
                        Text(assistant.recordingTranscript).font(AnyanwuTypography.body).lineSpacing(4)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 12) {
                        if assistant.isStartingRecording { ProgressView().controlSize(.small) }
                        else { RecordingWaveform(level: assistant.recorder.level) }
                        Text(assistant.isStartingRecording
                             ? language.text("Microphone…", "Igwe okwu…")
                             : language.text("Listening…", "A na-ege ntị…"))
                            .font(AnyanwuTypography.sans(13, relativeTo: .footnote)).foregroundStyle(AnyanwuTheme.muted)
                        if !assistant.isStartingRecording {
                            Text("\(Int(assistant.recorder.elapsed))s").font(AnyanwuTypography.caption2).monospacedDigit()
                                .foregroundStyle(AnyanwuTheme.muted)
                        }
                    }
                }
                .foregroundStyle(AnyanwuTheme.ink)
                .padding(.horizontal, 17).padding(.vertical, 14)
                .background(AnyanwuTheme.pale, in: UnevenRoundedRectangle(topLeadingRadius: 20, bottomLeadingRadius: 20, bottomTrailingRadius: 6, topTrailingRadius: 20))
            }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .accessibilityIdentifier("recording-transcript")
    }
}
