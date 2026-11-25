import XCTest

final class AnyanwuUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["-anyanwu-ui-testing", "-anyanwu-ui-language", "en"]
        app.launch()
    }

    func testMinimalHomeAndLanguageSelection() {
        XCTAssertTrue(app.staticTexts["Hello."].waitForExistence(timeout: 8))
        XCTAssertFalse(app.staticTexts["Make yourself heard."].exists)
        XCTAssertFalse(app.staticTexts["A VOICE THAT FEELS LIKE HOME"].exists)
        XCTAssertFalse(app.textFields["message-input"].exists)
        XCTAssertTrue(app.buttons["record-button"].exists)
        attachScreenshot("01-Home")
        app.buttons["app-language"].tap()
        let igbo = app.buttons["language-ig"].exists ? app.buttons["language-ig"] : app.buttons["Igbo"]
        igbo.tap()
        XCTAssertTrue(app.staticTexts["Ndewo."].waitForExistence(timeout: 5))
        app.buttons["tab-settings"].tap()
        XCTAssertTrue(app.staticTexts["Ntọala"].waitForExistence(timeout: 5))
        attachScreenshot("02-Igbo-settings")
    }

    func testHomeAndPreparedConversation() {
        app.buttons["open-composer"].tap()
        let input = app.textFields["message-input"].exists ? app.textFields["message-input"] : app.textViews["message-input"]
        input.tap()
        input.typeText("Hello")
        app.buttons["send-message"].tap()
        XCTAssertTrue(app.staticTexts["PREVIEW"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Ndewo! Kedu ka ị mere?"].exists)
        attachScreenshot("02-Conversation")
        app.buttons["tab-conversations"].tap()
        XCTAssertTrue(app.buttons.containing(NSPredicate(format: "label CONTAINS %@", "Hello")).firstMatch.exists)
    }

    func testTextComposition() {
        app.buttons["open-composer"].tap()
        let input = app.textFields["message-input"].exists ? app.textFields["message-input"] : app.textViews["message-input"]
        input.tap()
        input.typeText("How do I say thank you in Igbo?")
        app.buttons["send-message"].tap()
        XCTAssertTrue(app.staticTexts["Daalụ!"].waitForExistence(timeout: 5))
    }

    func testSettingsRejectsInsecureRemoteService() {
        app.buttons["tab-settings"].tap()
        app.segmentedControls["mode-picker"].buttons["Connected"].tap()
        let server = app.textFields["server-url"]
        server.tap()
        server.typeText("http://insecure.example")
        app.swipeUp()
        app.swipeUp()
        app.buttons["save-settings"].tap()
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "Enter an HTTPS service address")).firstMatch.waitForExistence(timeout: 5))
        attachScreenshot("03-Settings")
    }

    func testRecordingProducesPlayableLocalClip() {
        addUIInterruptionMonitor(withDescription: "Microphone permission") { alert in
            if alert.buttons["Allow"].exists { alert.buttons["Allow"].tap(); return true }
            if alert.buttons["OK"].exists { alert.buttons["OK"].tap(); return true }
            return false
        }
        app.buttons["record-button"].tap()
        if app.alerts.firstMatch.waitForExistence(timeout: 2) {
            let alert = app.alerts.firstMatch
            if alert.buttons["Allow"].exists { alert.buttons["Allow"].tap() }
            else if alert.buttons["OK"].exists { alert.buttons["OK"].tap() }
        }
        let stop = app.buttons["record-button"]
        XCTAssertTrue(stop.waitForExistence(timeout: 5))
        let recording = NSPredicate(format: "label == %@", "Stop recording")
        expectation(for: recording, evaluatedWith: stop)
        waitForExpectations(timeout: 8)
        // Capture a meaningful clip after the permission-dependent start has completed.
        Thread.sleep(forTimeInterval: 1.3)
        stop.tap()
        XCTAssertTrue(app.buttons["play-recording"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "Connect a voice service")).firstMatch.exists)
        attachScreenshot("04-Voice-recording")
    }

    private func attachScreenshot(_ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
