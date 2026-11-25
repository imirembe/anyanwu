import SwiftUI

struct VoiceControls: View {
    @Environment(AssistantStore.self) private var assistant
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }

    var body: some View {
        VStack(spacing: 12) {
            if assistant.isTextEntryVisible && !assistant.isStartingRecording && !assistant.recorder.isRecording {
                ComposerView().padding(.bottom, 2)
            }
            HStack(spacing: 18) {
                Button {
                    assistant.isTextEntryVisible.toggle()
                } label: {
                    Image(systemName: assistant.isTextEntryVisible ? "keyboard.chevron.compact.down" : "keyboard")
                        .font(.system(size: 19, weight: .regular)).frame(width: 44, height: 44)
                }
                .buttonStyle(.plain).foregroundStyle(AnyanwuTheme.muted)
                .accessibilityLabel(language.text("Type a message", "Dee ozi"))
                .accessibilityIdentifier("open-composer")
                .disabled(assistant.recorder.isRecording || assistant.isStartingRecording)
                Spacer(minLength: 16)
                RecordButton()
                Spacer(minLength: 16)
                Group {
                    if assistant.recorder.isRecording || assistant.isStartingRecording {
                        Button { assistant.cancelRecording() } label: {
                            Image(systemName: "xmark").font(.system(size: 17, weight: .regular)).frame(width: 44, height: 44)
                        }.accessibilityLabel(language.text("Discard recording", "Hapụ ndekọ"))
                    } else if assistant.isWorking {
                        Button { assistant.cancelTurn() } label: {
                            Image(systemName: "xmark").font(.system(size: 17, weight: .regular)).frame(width: 44, height: 44)
                        }.accessibilityLabel(language.text("Cancel", "Kagbuo"))
                    } else if assistant.isPlayingSpeech {
                        Button { assistant.stopPlayback() } label: {
                            Image(systemName: "speaker.slash").font(.system(size: 18)).frame(width: 44, height: 44)
                        }.accessibilityLabel(language.text("Stop playback", "Kwụsị ụda"))
                    } else { Color.clear.frame(width: 44, height: 44).accessibilityHidden(true) }
                }.buttonStyle(.plain).foregroundStyle(AnyanwuTheme.muted)
            }.frame(height: 72)
        }
        .padding(.horizontal, 28).padding(.top, 12).padding(.bottom, 14)
        .background(AnyanwuTheme.paper)
        .overlay(alignment: .top) {
            Rectangle().fill(AnyanwuTheme.line.opacity(0.75)).frame(height: 0.7).padding(.horizontal, 24)
        }
    }
}

struct RecordButton: View {
    @Environment(AssistantStore.self) private var assistant
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    private var active: Bool { assistant.recorder.isRecording }

    var body: some View {
        Button { assistant.toggleRecording() } label: {
            ZStack {
                Circle().fill(active ? AnyanwuTheme.pale : .white.opacity(0.75))
                Circle().stroke(active ? AnyanwuTheme.forest.opacity(0.3) : AnyanwuTheme.line, lineWidth: 1)
                if assistant.isWorking || assistant.isStartingRecording || (assistant.settings.mode == .apple && assistant.appleAssistant.isInstalling) {
                    ProgressView().tint(AnyanwuTheme.forest)
                } else {
                    Image(systemName: active ? "stop.fill" : "mic.fill")
                        .font(.system(size: active ? 18 : 22, weight: .medium)).foregroundStyle(AnyanwuTheme.forest)
                }
            }
            .frame(width: 60, height: 60)
            .overlay {
                if active {
                    Circle().stroke(AnyanwuTheme.forest.opacity(0.12), lineWidth: 2)
                        .scaleEffect(reduceMotion ? 1.1 : 1.08 + assistant.recorder.level * 0.15)
                        .animation(.easeOut(duration: 0.12), value: assistant.recorder.level)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(assistant.isWorking || assistant.isStartingRecording || assistant.isPreparingStorage || assistant.isLoadingConversation)
        .accessibilityIdentifier("record-button")
        .accessibilityLabel(active ? language.text("Stop recording", "Kwụsị ndekọ") : language.text("Record a voice message", "Dekọọ ozi olu"))
        .accessibilityHint(language.text("Speak in Igbo or English. Anyanwụ replies in Igbo.", "Kwuo okwu n'Igbo ma ọ bụ Bekee. Anyanwụ na-aza n'Igbo."))
        .sensoryFeedback(.impact(weight: .light), trigger: active)
    }
}

struct RecordingWaveform: View {
    var level: Double
    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<9, id: \.self) { index in
                let envelope = 1 - abs(Double(index - 4)) / 5
                Capsule().fill(AnyanwuTheme.forest.opacity(0.7))
                    .frame(width: 2.5, height: 5 + envelope * (5 + min(max(level, 0), 1) * 17))
            }
        }
        .frame(width: 50, height: 28)
        .animation(.easeOut(duration: 0.12), value: level)
        .accessibilityHidden(true)
    }
}
