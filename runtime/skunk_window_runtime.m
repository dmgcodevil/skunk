#import <Cocoa/Cocoa.h>
#include <ctype.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

@class SkunkCanvasView;
@class SkunkWindowDelegate;

typedef struct SkunkWindow {
    int32_t width;
    int32_t height;
    uint32_t *pixels;
    bool open;
    bool headless;
    double last_present_time;
    double delta_time;
    uint8_t key_down[256];
    NSWindow *ns_window;
    SkunkCanvasView *ns_view;
    SkunkWindowDelegate *delegate;
} SkunkWindow;

static double skunk_now_seconds(void) {
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t ticks = mach_absolute_time();
    double nanos = (double)ticks * (double)timebase.numer / (double)timebase.denom;
    return nanos / 1000000000.0;
}

static bool skunk_headless_enabled(void) {
    const char *value = getenv("SKUNK_WINDOW_HEADLESS");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void skunk_window_step_clock(SkunkWindow *window) {
    if (window == NULL) {
        return;
    }
    if (window->headless) {
        window->delta_time = 1.0 / 60.0;
        window->last_present_time = skunk_now_seconds();
        return;
    }
    double now = skunk_now_seconds();
    if (window->last_present_time <= 0.0) {
        window->delta_time = 1.0 / 60.0;
    } else {
        window->delta_time = now - window->last_present_time;
        if (window->delta_time <= 0.0) {
            window->delta_time = 1.0 / 60.0;
        }
    }
    window->last_present_time = now;
}

static void skunk_update_key_state(SkunkWindow *window, NSEvent *event, bool is_down) {
    if (window == NULL || event == nil) {
        return;
    }
    NSString *characters = [event charactersIgnoringModifiers];
    if (characters == nil || [characters length] == 0) {
        return;
    }
    unichar ch = [characters characterAtIndex:0];
    if (ch < 256) {
        int lowered = tolower((int)ch);
        if (lowered >= 0 && lowered < 256) {
            window->key_down[lowered] = is_down ? 1 : 0;
        }
    }
}

@interface SkunkWindowDelegate : NSObject <NSWindowDelegate> {
@public
    SkunkWindow *skunkWindow;
}
@end

@implementation SkunkWindowDelegate
- (BOOL)windowShouldClose:(id)sender {
    (void)sender;
    if (skunkWindow != NULL) {
        skunkWindow->open = false;
    }
    return YES;
}
@end

@interface SkunkCanvasView : NSView {
@public
    SkunkWindow *skunkWindow;
}
@end

@implementation SkunkCanvasView
- (BOOL)isFlipped {
    return YES;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (BOOL)canBecomeKeyView {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    skunk_update_key_state(skunkWindow, event, true);
}

- (void)keyUp:(NSEvent *)event {
    skunk_update_key_state(skunkWindow, event, false);
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    if (skunkWindow == NULL || skunkWindow->pixels == NULL) {
        return;
    }

    CGContextRef context = [[NSGraphicsContext currentContext] CGContext];
    CGContextSetInterpolationQuality(context, kCGInterpolationNone);

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(
        NULL,
        skunkWindow->pixels,
        (size_t)skunkWindow->width * (size_t)skunkWindow->height * sizeof(uint32_t),
        NULL
    );
    CGImageRef image = CGImageCreate(
        (size_t)skunkWindow->width,
        (size_t)skunkWindow->height,
        8,
        32,
        (size_t)skunkWindow->width * sizeof(uint32_t),
        color_space,
        kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little,
        provider,
        NULL,
        false,
        kCGRenderingIntentDefault
    );

    CGRect bounds = CGRectMake(0, 0, skunkWindow->width, skunkWindow->height);
    CGContextDrawImage(context, bounds, image);

    CGImageRelease(image);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(color_space);
}
@end

static void skunk_window_pump_events(SkunkWindow *window) {
    if (window == NULL || window->headless) {
        return;
    }

    @autoreleasepool {
        for (;;) {
            NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                untilDate:[NSDate distantPast]
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES];
            if (event == nil) {
                break;
            }
            [NSApp sendEvent:event];
        }
        [NSApp updateWindows];
    }
}

void *skunk_window_create(int32_t width, int32_t height, const char *title) {
    SkunkWindow *window = (SkunkWindow *)calloc(1, sizeof(SkunkWindow));
    if (window == NULL) {
        return NULL;
    }

    window->width = width > 0 ? width : 1;
    window->height = height > 0 ? height : 1;
    window->pixels = (uint32_t *)calloc(
        (size_t)window->width * (size_t)window->height,
        sizeof(uint32_t)
    );
    window->open = true;
    window->headless = skunk_headless_enabled();
    window->delta_time = 1.0 / 60.0;
    window->last_present_time = skunk_now_seconds();

    if (window->headless) {
        return window;
    }

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];

        NSRect frame = NSMakeRect(0, 0, window->width, window->height);
        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;

        window->ns_window = [[NSWindow alloc] initWithContentRect:frame
                                                        styleMask:style
                                                          backing:NSBackingStoreBuffered
                                                            defer:NO];
        window->ns_view = [[SkunkCanvasView alloc] initWithFrame:frame];
        window->ns_view->skunkWindow = window;
        window->delegate = [[SkunkWindowDelegate alloc] init];
        window->delegate->skunkWindow = window;

        NSString *ns_title = title != NULL ? [NSString stringWithUTF8String:title] : @"Skunk";
        [window->ns_window setTitle:ns_title];
        [window->ns_window setDelegate:window->delegate];
        [window->ns_window setReleasedWhenClosed:NO];
        [window->ns_window setContentView:window->ns_view];
        [window->ns_window center];
        [window->ns_window makeKeyAndOrderFront:nil];
        [window->ns_window makeFirstResponder:window->ns_view];
        [NSApp activateIgnoringOtherApps:YES];
    }

    skunk_window_pump_events(window);
    return window;
}

bool skunk_window_is_open(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return false;
    }
    skunk_window_pump_events(window);
    return window->open;
}

void skunk_window_poll(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return;
    }
    skunk_window_pump_events(window);
}

void skunk_window_clear(void *window_ptr, int32_t color) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL || window->pixels == NULL) {
        return;
    }
    uint32_t packed = (uint32_t)color;
    size_t count = (size_t)window->width * (size_t)window->height;
    for (size_t i = 0; i < count; ++i) {
        window->pixels[i] = packed;
    }
}

void skunk_window_draw_rect(
    void *window_ptr,
    double x,
    double y,
    double width,
    double height,
    int32_t color
) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL || window->pixels == NULL) {
        return;
    }

    int x0 = (int)llround(x);
    int y0 = (int)llround(y);
    int x1 = x0 + (int)llround(width);
    int y1 = y0 + (int)llround(height);

    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }
    if (x1 > window->width) {
        x1 = window->width;
    }
    if (y1 > window->height) {
        y1 = window->height;
    }
    if (x0 >= x1 || y0 >= y1) {
        return;
    }

    uint32_t packed = (uint32_t)color;
    for (int row = y0; row < y1; ++row) {
        size_t offset = (size_t)row * (size_t)window->width;
        for (int col = x0; col < x1; ++col) {
            window->pixels[offset + (size_t)col] = packed;
        }
    }
}

void skunk_window_present(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return;
    }

    if (!window->headless && window->ns_view != nil) {
        @autoreleasepool {
            [window->ns_view setNeedsDisplay:YES];
            [window->ns_view displayIfNeeded];
        }
    }

    skunk_window_pump_events(window);
    skunk_window_step_clock(window);
}

double skunk_window_delta_time(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return 0.0;
    }
    return window->delta_time;
}

void skunk_window_close(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return;
    }
    window->open = false;
    if (!window->headless && window->ns_window != nil) {
        @autoreleasepool {
            [window->ns_window orderOut:nil];
            [window->ns_window close];
        }
    }
}

void skunk_window_deinit(void *window_ptr) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL) {
        return;
    }

    if (!window->headless) {
        @autoreleasepool {
            if (window->ns_window != nil) {
                [window->ns_window setDelegate:nil];
            }
            if (window->delegate != nil) {
                [window->delegate release];
                window->delegate = nil;
            }
            if (window->ns_view != nil) {
                [window->ns_view release];
                window->ns_view = nil;
            }
            if (window->ns_window != nil) {
                [window->ns_window release];
                window->ns_window = nil;
            }
        }
    }

    free(window->pixels);
    free(window);
}

bool skunk_keyboard_is_down(void *window_ptr, uint16_t key) {
    SkunkWindow *window = (SkunkWindow *)window_ptr;
    if (window == NULL || key >= 256) {
        return false;
    }
    int lowered = tolower((int)key);
    if (lowered < 0 || lowered >= 256) {
        return false;
    }
    return window->key_down[lowered] != 0;
}
