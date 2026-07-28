"""An AppKit window with a CAMetalLayer, and the mouse.

The WebGPU side gets a window for free (``rendercanvas`` carries a GLFW
backend and hands us a configured surface). Metal has no such library in
this venv, so the window is hand-built: NSApplication + NSWindow + a
layer-hosting NSView whose layer is a CAMetalLayer, an NSTrackingArea for
mouse-moved events, and an NSTimer driving frames.

What the presenter actually costs, beyond the offscreen path:

  * ``layer.nextDrawable()`` replaces "the texture I own" -- it can
    return nil under memory pressure and must be re-requested every
    frame, and its texture must not be retained past ``present``.
  * ``drawableSize`` is in PIXELS and the view's bounds are in POINTS,
    so the backing scale factor has to be applied by hand. Nothing else
    in the pipeline cares: resolution never entered the shader.
  * ``cmd.presentDrawable_(drawable)`` before ``commit()`` is the whole
    of presentation.

``Engine.encode`` is called here EXACTLY as the offscreen verifier calls
it -- same method, same arguments, a different texture. That sharing is
the point; if the window path needed its own encode we would have two
programs to keep in agreement instead of one.

The frame loop waits on each command buffer (``waitUntilCompleted``)
because the slot buffers are written in place on unified memory and a
frame still reading them would see torn uniforms. A real presenter uses a
2-3 deep semaphore and rotating slot buffers; that is a scheduling
decision the encodable shape should be able to express and today cannot.

UNVERIFIED BY THE AGENT THAT WROTE IT: this file opens a window, which a
headless session cannot do. What IS verified headless is every line of
the drawing path -- ``demo/mouse_ripple.py --check-layer`` builds a real
CAMetalLayer, takes a real drawable, and runs the real ``Engine.encode``
into it, so only NSApp/NSWindow/tracking-area behaviour is untested.
"""

from __future__ import annotations

import time

import objc
from Cocoa import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSMakeRect,
    NSObject,
    NSRunLoop,
    NSRunLoopCommonModes,
    NSTimer,
    NSTrackingActiveInKeyWindow,
    NSTrackingArea,
    NSTrackingInVisibleRect,
    NSTrackingMouseMoved,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Quartz import CAMetalLayer


def make_layer(device, width: int, height: int, pixel_format, scale: float = 2.0):
    """A CAMetalLayer ready to hand out drawables. Split out of the window
    so the headless check can exercise it without an NSApp."""
    layer = CAMetalLayer.layer()
    layer.setDevice_(device)
    layer.setPixelFormat_(pixel_format)
    layer.setFramebufferOnly_(True)
    layer.setContentsScale_(scale)
    layer.setDrawableSize_((width * scale, height * scale))
    return layer


class RippleView(NSView):
    """Layer-hosting view: we assign the CAMetalLayer ourselves rather
    than letting AppKit make one, and we take mouse-moved events through
    a tracking area that follows the visible rect."""

    def initWithFrame_layer_mouse_(self, frame, layer, mouse):
        self = objc.super(RippleView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._mouse = mouse
        self.setLayer_(layer)  # set the layer BEFORE wantsLayer: layer-hosting
        self.setWantsLayer_(True)
        return self

    def acceptsFirstResponder(self):
        return True

    def updateTrackingAreas(self):
        for area in self.trackingAreas():
            self.removeTrackingArea_(area)
        opts = NSTrackingMouseMoved | NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect
        self.addTrackingArea_(
            NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(self.bounds(), opts, self, None)
        )
        objc.super(RippleView, self).updateTrackingAreas()

    def mouseMoved_(self, event):
        self._deliver(event)

    def mouseDragged_(self, event):
        self._deliver(event)

    @objc.python_method
    def _deliver(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        b = self.bounds()
        # AppKit's origin is BOTTOM-left; the demo's canonical cursor has
        # v = 0 at the TOP, so every backend flips into that convention.
        self._mouse.set(p.x / max(b.size.width, 1.0), 1.0 - p.y / max(b.size.height, 1.0))


class Ticker(NSObject):
    """NSTimer needs an ObjC target; this is it."""

    def initWithCallback_(self, cb):
        self = objc.super(Ticker, self).init()
        if self is None:
            return None
        self._cb = cb
        return self

    def tick_(self, timer):
        self._cb()


def run(low, mouse, on_frame, width=900, height=600, title="pdum"):
    import msl_glue

    rt = msl_glue.runtime()
    fmt = "bgra8unorm-srgb"
    engine = msl_glue.Engine(low, fmt, rt=rt)
    print(f"[metal] surface format {fmt}; one pipeline, {low.draw_count} vertices, no per-frame lowering")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    style = (
        NSWindowStyleMaskTitled
        | NSWindowStyleMaskClosable
        | NSWindowStyleMaskMiniaturizable
        | NSWindowStyleMaskResizable
    )
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(120, 120, width, height), style, NSBackingStoreBuffered, False
    )
    win.setTitle_(title)
    win.setAcceptsMouseMovedEvents_(True)

    layer = make_layer(rt.device, width, height, msl_glue._PIXEL[fmt], scale=1.0)
    view = RippleView.alloc().initWithFrame_layer_mouse_(NSMakeRect(0, 0, width, height), layer, mouse)
    win.setContentView_(view)
    win.makeFirstResponder_(view)
    win.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)

    state = {"n": 0, "t0": time.perf_counter()}

    def frame():
        # The backing scale is read EVERY frame, not once at setup: a
        # window that is not yet on screen reports 1.0 whatever display it
        # is destined for (verified headless), and dragging a window
        # between a Retina and a non-Retina display changes it live.
        scale = win.backingScaleFactor() or 1.0
        layer.setContentsScale_(scale)
        b = view.bounds()
        want = (b.size.width * scale, b.size.height * scale)
        if tuple(layer.drawableSize()) != want:
            layer.setDrawableSize_(want)  # resolution keys nothing: no recompile
        drawable = layer.nextDrawable()
        if drawable is None:
            return
        engine.update(on_frame())
        cmd = rt.queue.commandBuffer()
        engine.encode(cmd, drawable.texture(), clear=(0.02, 0.02, 0.03, 1.0))
        cmd.presentDrawable_(drawable)
        cmd.commit()
        cmd.waitUntilCompleted()
        state["n"] += 1
        dt = time.perf_counter() - state["t0"]
        if dt >= 2.0:
            print(f"{state['n'] / dt:6.1f} fps")
            state["t0"], state["n"] = time.perf_counter(), 0

    # The timer goes in on COMMON modes so frames keep coming while the
    # user is dragging the title bar or resizing -- the default mode
    # stalls the loop for the whole duration of a modal tracking session,
    # which for a mouse-driven demo is exactly the wrong moment to stop.
    ticker = Ticker.alloc().initWithCallback_(frame)
    timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(1.0 / 120.0, ticker, "tick:", None, True)
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
    app.run()
