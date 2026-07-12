// jsdom smoke tests over the generated fixtures: structure, tooltips, isolation.
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { JSDOM } from "jsdom";

const load = (name) =>
  new JSDOM(readFileSync(new URL(`./fixtures/${name}.html`, import.meta.url), "utf8")).window.document;

test("handle card: pills, collapsible source, scoped root", () => {
  const doc = load("handle");
  assert.ok(doc.querySelector(".pdum-viz .pv-card"));
  assert.ok(doc.querySelectorAll(".pv-pill").length >= 2);
  assert.ok(doc.querySelector("details > summary"));
  assert.equal(doc.querySelectorAll("script").length, 0); // static like a PNG
});

test("region widget: hover tooltips carry type + provenance", () => {
  const doc = load("region");
  const tips = [...doc.querySelectorAll(".pv-tip")].map((el) => el.getAttribute("data-tip"));
  assert.ok(tips.some((t) => t && t.includes("f64") && t.includes("art.py:7")));
});

test("pipeline composes stage fragments with one style block", () => {
  const doc = load("pipeline");
  assert.equal(doc.querySelectorAll("style").length, 1);
  assert.ok(doc.querySelectorAll(".pv-chip").length >= 2);
});
