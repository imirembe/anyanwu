import SwiftUI

/// Explicit bundled faces keep Igbo letters and combining tone marks in the
/// same family. Custom fonts scale once with the specified Dynamic Type style.
enum AnyanwuTypography {
    static let body = sans(16)
    static let caption = sans(12, relativeTo: .caption)
    static let caption2 = sans(11, relativeTo: .caption2)
    static let subheadline = sans(15, relativeTo: .subheadline)

    static func sans(_ size: CGFloat, weight: Font.Weight = .regular,
                     relativeTo style: Font.TextStyle = .body) -> Font {
        .custom("NotoSans-\(face(for: weight))", size: size, relativeTo: style)
    }

    static func serif(_ size: CGFloat, weight: Font.Weight = .regular,
                      relativeTo style: Font.TextStyle = .title) -> Font {
        .custom("NotoSerif-\(face(for: weight))", size: size, relativeTo: style)
    }

    private static func face(for weight: Font.Weight) -> String {
        switch weight {
        case .medium: "Medium"
        case .semibold: "SemiBold"
        case .bold: "Bold"
        case .light: "Light"
        default: "Regular"
        }
    }
}
