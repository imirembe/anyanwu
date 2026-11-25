import SwiftUI

struct HistoryView: View {
    @Environment(AssistantStore.self) private var assistant
    @State private var deleting: ConversationSummary?
    private var language: InterfaceLanguage { assistant.settings.interfaceLanguage }
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(language.text("Conversations", "Mkparịta ụka"))
                .font(AnyanwuTypography.serif(32, relativeTo: .largeTitle)).foregroundStyle(AnyanwuTheme.forest).padding(.top, 16)
            if !assistant.settings.saveHistory {
                Text(language.text("History is off. Only the current conversation stays in memory.", "Agbanyụọla akụkọ. Naanị mkparịta ụka ugbu a dị na ebe nchekwa nwa oge."))
                    .font(AnyanwuTypography.caption).foregroundStyle(AnyanwuTheme.muted).padding(.top, 8)
            }
            if assistant.isPreparingStorage || assistant.isLoadingHistory {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if assistant.conversations.isEmpty {
                VStack(spacing: 20) {
                    Image(systemName: "bubble.left.and.bubble.right").font(.system(size: 37, weight: .ultraLight))
                    Text(language.text("No conversations yet.", "Enweghị mkparịta ụka."))
                        .font(AnyanwuTypography.serif(22, relativeTo: .title2))
                    Button(language.text("Start a conversation", "Malite mkparịta ụka")) { assistant.newConversation() }
                        .font(AnyanwuTypography.sans(14, weight: .medium, relativeTo: .subheadline)).padding(.horizontal, 22).padding(.vertical, 14)
                        .background(AnyanwuTheme.forest, in: Capsule()).foregroundStyle(AnyanwuTheme.paper)
                }.frame(maxWidth: .infinity, maxHeight: .infinity).foregroundStyle(AnyanwuTheme.forest)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if assistant.hasNewerConversations {
                            Button(language.text("Newer conversations", "Mkparịta ụka ọhụrụ karịa")) { assistant.showNewerConversations() }
                        }
                        ForEach(assistant.conversations) { conversation in
                            HStack(spacing: 10) {
                                Button { assistant.open(conversation) } label: {
                                    VStack(alignment: .leading, spacing: 10) {
                                        Text(conversation.title(in: language)).font(AnyanwuTypography.body)
                                            .foregroundStyle(AnyanwuTheme.forest).lineLimit(2)
                                        HStack {
                                            Text(conversation.updatedAt, format: .dateTime.month(.abbreviated).day())
                                            Text("·")
                                            Text(language.text("\(conversation.messageCount) messages", "Ozi \(conversation.messageCount)"))
                                        }.font(AnyanwuTypography.caption2).foregroundStyle(AnyanwuTheme.muted)
                                    }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
                                }.buttonStyle(.plain)
                                Button { deleting = conversation } label: {
                                    Image(systemName: "trash").font(.system(size: 14)).foregroundStyle(AnyanwuTheme.muted)
                                        .frame(width: 44, height: 44)
                                }.accessibilityLabel(language.text("Delete conversation", "Hichapụ mkparịta ụka"))
                            }.padding(16).background(.white.opacity(0.65), in: RoundedRectangle(cornerRadius: 19))
                        }
                        if assistant.hasOlderConversations {
                            Button(language.text("Older conversations", "Mkparịta ụka ochie karịa")) { assistant.showOlderConversations() }
                        }
                    }.padding(.top, 24).padding(.bottom, 20)
                }.scrollIndicators(.hidden)
            }
            if let error = assistant.error { NoticeView(text: error, isError: true) { assistant.error = nil } }
        }
        .padding(.horizontal, 24)
        .confirmationDialog(language.text("Delete this conversation and its recordings?", "Hichapụ mkparịta ụka a na ndekọ ya?"), isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }
        ), titleVisibility: .visible) {
            Button(language.text("Delete", "Hichapụ"), role: .destructive) {
                if let deleting { assistant.deleteConversation(deleting.id) }
                deleting = nil
            }
            Button(language.text("Cancel", "Kagbuo"), role: .cancel) { deleting = nil }
        }
    }
}
