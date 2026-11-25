import SwiftUI

enum AnyanwuTheme {
    static let paper = Color(hex: 0xF6F3EC)
    static let forest = Color(hex: 0x193E32)
    static let muted = Color(hex: 0x747B6F)
    static let gold = Color(hex: 0xBE8E45)
    static let line = Color(hex: 0xDFE2D7)
    static let pale = Color(hex: 0xEAEDE3)
    static let ink = Color(hex: 0x24382E)
}

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255, opacity: 1)
    }
}

struct AnyanwuMark: View {
    var size: CGFloat = 34
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.28).fill(AnyanwuTheme.forest)
            Image(systemName: "sun.max.fill")
                .font(.system(size: size * 0.56, weight: .medium))
                .foregroundStyle(AnyanwuTheme.gold)
        }.frame(width: size, height: size).accessibilityHidden(true)
    }
}

struct Eyebrow: View {
    let title: String
    var body: some View {
        Text(title).font(AnyanwuTypography.sans(10, weight: .semibold, relativeTo: .caption2))
            .tracking(1.6).foregroundStyle(AnyanwuTheme.muted)
    }
}

struct NoticeView: View {
    @Environment(AssistantStore.self) private var assistant
    let text: String
    var isError = false
    var dismiss: (() -> Void)?
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: isError ? "exclamationmark.circle" : "info.circle")
            Text(text).font(AnyanwuTypography.subheadline).fixedSize(horizontal: false, vertical: true)
            if let dismiss {
                Button(action: dismiss) { Image(systemName: "xmark").font(.caption.bold()) }
                    .accessibilityLabel(assistant.settings.interfaceLanguage.text("Dismiss notice", "Mechie ọkwa"))
            }
        }
        .foregroundStyle(isError ? Color(hex: 0x853D30) : AnyanwuTheme.forest)
        .padding(14).frame(maxWidth: .infinity, alignment: .leading)
        .background(isError ? Color(hex: 0xF4E7DF) : AnyanwuTheme.pale, in: RoundedRectangle(cornerRadius: 16))
    }
}
