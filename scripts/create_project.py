#!/usr/bin/env python3
"""Generate the dependency-free Xcode project, including shared test scheme."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
objects = {}


def identifier(name):
    return hashlib.sha1(("anyanwu/" + name).encode()).hexdigest()[:24].upper()


def add(object_name, isa, **values):
    key = identifier(object_name)
    objects[key] = {"isa": isa, **values}
    return key


def encode(value, indent=0):
    pad = "\t" * indent
    if isinstance(value, dict):
        return "{\n" + "".join(f"{pad}\t{json.dumps(k)} = {encode(v, indent + 1)};\n" for k, v in value.items()) + pad + "}"
    if isinstance(value, list):
        return "(\n" + "".join(f"{pad}\t{encode(v, indent + 1)},\n" for v in value) + pad + ")"
    return json.dumps(str(value))


app_id, tests_id, ui_id = (identifier(name) for name in ("app", "tests", "uitests"))
app_sources = add("app-files", "PBXFileSystemSynchronizedRootGroup", path="Sources/Anyanwu", sourceTree="<group>")
test_sources = add("test-files", "PBXFileSystemSynchronizedRootGroup", path="Tests/AnyanwuTests", sourceTree="<group>")
ui_sources = add("ui-files", "PBXFileSystemSynchronizedRootGroup", path="Tests/AnyanwuUITests", sourceTree="<group>")
info = add("info", "PBXFileReference", lastKnownFileType="text.plist.xml", path="Configuration/Info.plist", sourceTree="<group>")
signing = add("signing-config", "PBXFileReference", lastKnownFileType="text.xcconfig", path="Configuration/Signing.xcconfig", sourceTree="<group>")
products = []
for name, filename, kind in (("app", "Anyanwu.app", "wrapper.application"), ("tests", "AnyanwuTests.xctest", "wrapper.cfbundle"), ("uitests", "AnyanwuUITests.xctest", "wrapper.cfbundle")):
    products.append(add(name + "-product", "PBXFileReference", explicitFileType=kind, includeInIndex=0, path=filename, sourceTree="BUILT_PRODUCTS_DIR"))
products_group = add("products", "PBXGroup", children=products, name="Products", sourceTree="<group>")
main_group = add("main-group", "PBXGroup", children=[app_sources, test_sources, ui_sources, info, signing, products_group], sourceTree="<group>")


def configurations(name, settings):
    configs = []
    for mode in ("Debug", "Release"):
        merged = dict(settings)
        if mode == "Debug":
            merged.update(SWIFT_OPTIMIZATION_LEVEL="-Onone", GCC_OPTIMIZATION_LEVEL="0", ENABLE_TESTABILITY="YES", SWIFT_ACTIVE_COMPILATION_CONDITIONS="DEBUG $(inherited)")
        else:
            merged.update(SWIFT_COMPILATION_MODE="wholemodule", SWIFT_OPTIMIZATION_LEVEL="-O")
        extra = {"baseConfigurationReference": signing} if name == "project" else {}
        configs.append(add(f"{name}-{mode}", "XCBuildConfiguration", name=mode, buildSettings=merged, **extra))
    return add(name + "-configs", "XCConfigurationList", buildConfigurations=configs, defaultConfigurationIsVisible=0, defaultConfigurationName="Release")


project_configs = configurations("project", {
    "SDKROOT": "iphoneos", "IPHONEOS_DEPLOYMENT_TARGET": "18.0", "SWIFT_VERSION": "5.0",
    "CLANG_ENABLE_MODULES": "YES", "CLANG_ENABLE_OBJC_ARC": "YES", "ENABLE_STRICT_OBJC_MSGSEND": "YES",
    "GCC_C_LANGUAGE_STANDARD": "gnu17", "CLANG_CXX_LANGUAGE_STANDARD": "gnu++20", "DEBUG_INFORMATION_FORMAT": "dwarf",
    "SWIFT_STRICT_CONCURRENCY": "targeted", "ENABLE_USER_SCRIPT_SANDBOXING": "YES",
})
common = {"CODE_SIGN_STYLE": "Automatic", "DEVELOPMENT_TEAM": "$(ANYANWU_DEVELOPMENT_TEAM)", "TARGETED_DEVICE_FAMILY": "1", "SWIFT_EMIT_LOC_STRINGS": "YES"}
app_configs = configurations("app", dict(common, PRODUCT_NAME="Anyanwu", PRODUCT_BUNDLE_IDENTIFIER="com.ndichie.assistant",
    INFOPLIST_FILE="Configuration/Info.plist", GENERATE_INFOPLIST_FILE="NO", MARKETING_VERSION="0.1.0", CURRENT_PROJECT_VERSION="2",
    ASSETCATALOG_COMPILER_APPICON_NAME="AppIcon", LD_RUNPATH_SEARCH_PATHS="$(inherited) @executable_path/Frameworks"))
test_configs = configurations("tests", dict(common, PRODUCT_NAME="AnyanwuTests", PRODUCT_BUNDLE_IDENTIFIER="com.ndichie.assistant.tests",
    GENERATE_INFOPLIST_FILE="YES", TEST_HOST="$(BUILT_PRODUCTS_DIR)/Anyanwu.app/Anyanwu", BUNDLE_LOADER="$(TEST_HOST)"))
ui_configs = configurations("uitests", dict(common, PRODUCT_NAME="AnyanwuUITests", PRODUCT_BUNDLE_IDENTIFIER="com.ndichie.assistant.uitests",
    GENERATE_INFOPLIST_FILE="YES", TEST_TARGET_NAME="Anyanwu"))

for name, target_id, sources, configs, product, product_type in (
    ("app", app_id, app_sources, app_configs, products[0], "com.apple.product-type.application"),
    ("tests", tests_id, test_sources, test_configs, products[1], "com.apple.product-type.bundle.unit-test"),
    ("uitests", ui_id, ui_sources, ui_configs, products[2], "com.apple.product-type.bundle.ui-testing"),
):
    phases = []
    for phase in ("Sources", "Frameworks", "Resources"):
        phases.append(add(name + "-" + phase, "PBX" + phase + "BuildPhase", buildActionMask=2147483647, files=[], runOnlyForDeploymentPostprocessing=0))
    dependencies = []
    if name != "app":
        proxy = add(name + "-proxy", "PBXContainerItemProxy", containerPortal=identifier("project"), proxyType=1, remoteGlobalIDString=app_id, remoteInfo="Anyanwu")
        dependencies.append(add(name + "-dependency", "PBXTargetDependency", target=app_id, targetProxy=proxy))
    add(name, "PBXNativeTarget", buildConfigurationList=configs, buildPhases=phases, buildRules=[], dependencies=dependencies,
        fileSystemSynchronizedGroups=[sources], name={"app": "Anyanwu", "tests": "AnyanwuTests", "uitests": "AnyanwuUITests"}[name],
        productName={"app": "Anyanwu", "tests": "AnyanwuTests", "uitests": "AnyanwuUITests"}[name], productReference=product, productType=product_type)

project = add("project", "PBXProject", attributes={"BuildIndependentTargetsInParallel": "YES", "LastSwiftUpdateCheck": "2700", "LastUpgradeCheck": "2700",
    "TargetAttributes": {app_id: {"CreatedOnToolsVersion": "27.1"}, tests_id: {"CreatedOnToolsVersion": "27.1", "TestTargetID": app_id}, ui_id: {"CreatedOnToolsVersion": "27.1", "TestTargetID": app_id}}},
    buildConfigurationList=project_configs, developmentRegion="en", hasScannedForEncodings=0, knownRegions=["en", "Base", "ig"],
    mainGroup=main_group, minimizedProjectReferenceProxies=1, preferredProjectObjectVersion=77, productRefGroup=products_group,
    projectDirPath="", projectRoot="", targets=[app_id, tests_id, ui_id])
project_dir = ROOT / "Anyanwu.xcodeproj"
project_dir.mkdir(exist_ok=True)
(project_dir / "project.pbxproj").write_text("// !$*UTF8*$!\n" + encode({"archiveVersion": 1, "classes": {}, "objectVersion": 77, "objects": objects, "rootObject": project}) + "\n")


def reference(target, name, buildable):
    return f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="{buildable}" BlueprintName="{name}" ReferencedContainer="container:Anyanwu.xcodeproj"/>'


scheme = f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2700" version="1.3">
  <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries>
    <BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{reference(app_id, "Anyanwu", "Anyanwu.app")}</BuildActionEntry>
    <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="YES">{reference(tests_id, "AnyanwuTests", "AnyanwuTests.xctest")}</BuildActionEntry>
    <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="YES">{reference(ui_id, "AnyanwuUITests", "AnyanwuUITests.xctest")}</BuildActionEntry>
  </BuildActionEntries></BuildAction>
  <TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables>
    <TestableReference skipped="NO" parallelizable="NO">{reference(tests_id, "AnyanwuTests", "AnyanwuTests.xctest")}</TestableReference>
    <TestableReference skipped="NO" parallelizable="NO">{reference(ui_id, "AnyanwuUITests", "AnyanwuUITests.xctest")}</TestableReference>
  </Testables></TestAction>
  <LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference(app_id, "Anyanwu", "Anyanwu.app")}</BuildableProductRunnable></LaunchAction>
  <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference(app_id, "Anyanwu", "Anyanwu.app")}</BuildableProductRunnable></ProfileAction>
  <AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>
'''
scheme_dir = project_dir / "xcshareddata/xcschemes"
scheme_dir.mkdir(parents=True, exist_ok=True)
(scheme_dir / "Anyanwu.xcscheme").write_text(scheme)
print("Created Anyanwu.xcodeproj and shared Anyanwu scheme")
