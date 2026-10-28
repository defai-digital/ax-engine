// A headless Sparkle client used only by the opt-in installation integration test.
#import <AppKit/AppKit.h>
#import <Sparkle/Sparkle.h>

static void record(NSString *event) {
    NSString *directory = NSBundle.mainBundle.infoDictionary[@"AXProbeDirectory"];
    NSString *line = [event stringByAppendingString:@"\n"];
    NSFileHandle *file = [NSFileHandle fileHandleForWritingAtPath:[directory stringByAppendingPathComponent:@"events.txt"]];
    [file seekToEndOfFile];
    [file writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
    [file closeFile];
}

@interface Probe : NSObject <NSApplicationDelegate, SPUUserDriver, SPUUpdaterDelegate>
@property SPUUpdater *updater;
@end
@implementation Probe
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    if ([NSBundle.mainBundle.infoDictionary[@"CFBundleVersion"] isEqual:@"2.0.0"]) {
        record(@"relaunched 2.0.0"); exit(0);
    }
    self.updater = [[SPUUpdater alloc] initWithHostBundle:NSBundle.mainBundle applicationBundle:NSBundle.mainBundle userDriver:self delegate:self];
    NSError *error = nil;
    if (![self.updater startUpdater:&error]) { record([@"error " stringByAppendingString:error.description]); exit(1); }
    [self.updater checkForUpdates];
}
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    record(@"graceful shutdown requested");
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 10), dispatch_get_main_queue(), ^{
        record(@"cleanup finished"); [sender replyToApplicationShouldTerminate:YES];
    });
    return NSTerminateLater;
}
- (void)showUpdatePermissionRequest:(SPUUpdatePermissionRequest *)request reply:(void (^)(SUUpdatePermissionResponse *))reply { reply([[SUUpdatePermissionResponse alloc] initWithAutomaticUpdateChecks:NO sendSystemProfile:NO]); }
- (void)showUserInitiatedUpdateCheckWithCancellation:(void (^)(void))cancellation { record(@"checking"); }
- (void)showUpdateFoundWithAppcastItem:(SUAppcastItem *)item state:(SPUUserUpdateState *)state reply:(void (^)(SPUUserUpdateChoice))reply { record([@"found " stringByAppendingString:item.versionString]); if ([NSBundle.mainBundle.infoDictionary[@"AXProbeMode"] isEqual:@"cancel"]) { record(@"cancelled"); reply(SPUUserUpdateChoiceDismiss); exit(0); } reply(SPUUserUpdateChoiceInstall); }
- (void)showUpdateReleaseNotesWithDownloadData:(SPUDownloadData *)data {}
- (void)showUpdateReleaseNotesFailedToDownloadWithError:(NSError *)error {}
- (void)showUpdateNotFoundWithError:(NSError *)error acknowledgement:(void (^)(void))acknowledgement { record(@"up to date"); acknowledgement(); exit(0); }
- (void)showUpdaterError:(NSError *)error acknowledgement:(void (^)(void))acknowledgement { record([@"error " stringByAppendingString:error.description]); acknowledgement(); exit(1); }
- (void)showDownloadInitiatedWithCancellation:(void (^)(void))cancellation { record(@"downloading"); }
- (void)showDownloadDidReceiveExpectedContentLength:(uint64_t)length {}
- (void)showDownloadDidReceiveDataOfLength:(uint64_t)length {}
- (void)showDownloadDidStartExtractingUpdate { record(@"extracting"); }
- (void)showExtractionReceivedProgress:(double)progress {}
- (void)showReadyToInstallAndRelaunch:(void (^)(SPUUserUpdateChoice))reply { record(@"ready to install"); reply(SPUUserUpdateChoiceInstall); }
- (void)showInstallingUpdateWithApplicationTerminated:(BOOL)terminated retryTerminatingApplication:(void (^)(void))retry { record(@"installing"); }
- (void)showUpdateInstalledAndRelaunched:(BOOL)relaunched acknowledgement:(void (^)(void))acknowledgement { record(@"installed"); acknowledgement(); }
- (void)dismissUpdateInstallation {}
@end

int main(void) {
    @autoreleasepool {
        NSApplication *application = NSApplication.sharedApplication;
        [application setActivationPolicy:NSApplicationActivationPolicyProhibited];
        Probe *probe = [Probe new]; application.delegate = probe;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 90 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ record(@"timeout"); exit(2); });
        [application run];
    }
    return 0;
}
