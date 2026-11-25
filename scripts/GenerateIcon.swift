import AppKit
import CoreGraphics
import Foundation
import ImageIO

// Prepare the supplied artwork; never replace it with a generated monogram.
guard CommandLine.arguments.count == 2 else {
    fatalError("Usage: swift scripts/GenerateIcon.swift <output.png>")
}
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let sourceURL = root.appendingPathComponent("app Icon.png")
guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fatalError("Could not read app Icon.png")
}
let size = 1024
let colorSpace = CGColorSpaceCreateDeviceRGB()
let context = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                        bytesPerRow: size * 4, space: colorSpace,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
let bounds = CGRect(x: 0, y: 0, width: CGFloat(size), height: CGFloat(size))
context.fill(bounds)
context.interpolationQuality = .high
context.draw(image, in: bounds)
let representation = NSBitmapImageRep(cgImage: context.makeImage()!)
try representation.representation(using: .png, properties: [:])!
    .write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
print("Prepared Anyanwụ app icon from app Icon.png")
