---
name: Robin
description: "Preflight — the dark-web OSINT working screen as a flight-control board: flat near-black panels, labeled GO/NO-GO telemetry, one action blue, and system mono for every measured number."
colors:
  # grounds — one near-black family, stepped by surface role
  ground: "#0B0D10"
  band: "#0E1216"
  well: "#0F1319"
  surface: "#15191F"
  # hairlines — panel border, inner divider, widget stroke
  line: "#232830"
  line-faint: "#1D222A"
  control-line: "#2A3038"
  # text steps, brightest first
  ink: "#E8EAF0"
  control-ink: "#C9D0DC"
  readout: "#B8BFCC"
  heading: "#9AA2B2"
  dim: "#8C94A4"
  faint: "#7C8494"
  # on-color inks for solid status fills
  on-accent: "#06121F"
  on-go: "#06170E"
  on-down: "#1F0505"
  # the one action accent, plus its hover
  action: "#4C8DFF"
  action-hover: "#3D7FF0"
  # exactly three status hues, with their 8% tints and tinted hairlines
  go: "#3ED598"
  warn: "#FFB020"
  down: "#FF5A5A"
  go-tint: "rgba(62, 213, 152, 0.08)"
  warn-tint: "rgba(255, 176, 32, 0.08)"
  down-tint: "rgba(255, 90, 90, 0.08)"
  live-tint: "rgba(76, 141, 255, 0.10)"
  go-line: "#1E3A2C"
  warn-line: "#4A3A14"
  down-line: "#43222A"
typography:
  brand:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.18em"
  value:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 500
    lineHeight: 1.3
  body:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "10.5px"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.18em"
  data:
    fontFamily: "ui-monospace, \"SF Mono\", SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.4
    fontFeature: "tnum"
  query:
    fontFamily: "ui-monospace, \"SF Mono\", SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.3
rounded:
  corner: "2px"
  round: "50%"
  scrollbar-thumb: "6px"
spacing:
  micro: "3px"
  xs: "10px"
  sm: "12px"
  md: "14px"
  lg: "16px"
  xl: "24px"
components:
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.corner}"
  panel-header:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.heading}"
    height: "38px"
  checklist-row:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    height: "50px"
    padding: "10px 16px"
  pill-go:
    backgroundColor: "{colors.go}"
    textColor: "{colors.on-go}"
    rounded: "{rounded.corner}"
    padding: "5px 10px"
  pill-nogo:
    backgroundColor: "{colors.down}"
    textColor: "{colors.on-down}"
    rounded: "{rounded.corner}"
    padding: "5px 10px"
  pill-amber:
    backgroundColor: "{colors.warn-tint}"
    textColor: "{colors.warn}"
    rounded: "{rounded.corner}"
    padding: "5px 10px"
  pill-neutral:
    backgroundColor: "{colors.well}"
    textColor: "{colors.dim}"
    rounded: "{rounded.corner}"
    padding: "5px 10px"
  chip-up:
    backgroundColor: "{colors.go-tint}"
    textColor: "{colors.go}"
    rounded: "{rounded.corner}"
    padding: "4px 8px"
  chip-down:
    backgroundColor: "{colors.down-tint}"
    textColor: "{colors.down}"
    rounded: "{rounded.corner}"
    padding: "4px 8px"
  stage-cell:
    backgroundColor: "transparent"
    textColor: "{colors.dim}"
    padding: "10px 16px"
  stage-cell-live:
    backgroundColor: "{colors.live-tint}"
    textColor: "{colors.ink}"
  button-launch:
    backgroundColor: "{colors.action}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.corner}"
    height: "54px"
    width: "100%"
  button-launch-hover:
    backgroundColor: "{colors.action-hover}"
    textColor: "{colors.on-accent}"
  button-utility:
    backgroundColor: "{colors.well}"
    textColor: "{colors.control-ink}"
    rounded: "{rounded.corner}"
    height: "40px"
    padding: "0 14px"
  button-utility-hover:
    backgroundColor: "{colors.well}"
    textColor: "{colors.ink}"
  input-query:
    backgroundColor: "{colors.well}"
    textColor: "{colors.ink}"
    rounded: "{rounded.corner}"
    height: "52px"
    padding: "0 16px"
  log-entry:
    backgroundColor: "transparent"
    textColor: "{colors.readout}"
    padding: "9px 16px"
  source-row:
    backgroundColor: "transparent"
    textColor: "{colors.readout}"
    height: "32px"
  engine-row:
    backgroundColor: "transparent"
    textColor: "{colors.readout}"
    height: "30px"
  tor-chip:
    backgroundColor: "{colors.well}"
    textColor: "{colors.dim}"
    rounded: "{rounded.corner}"
    padding: "7px 12px"
---

# Design System: Robin

## Overview

**Creative North Star: "Preflight"**

The working screen is a flight-control board. Robin's real state — Tor circuit, configured providers, armed model, the sixteen engines — reads before anything launches, and the six-stage pipeline runs as a mission phase strip paced by each stage's measured duration. Nothing on the screen is decorative: every panel, label, and number is an instrument reading or a control. The world refuses the category default (a bare search box on a void, health buried behind a sidebar button): state first, launch second, record last.

In material terms the console is quiet and cold: a near-black deck, flat panels separated by 1px hairlines with no shadows anywhere, and one blue that belongs to things you can press and to the live channel. Status color is used like annunciator lighting — exactly three hues, each always attached to a labeled pill (GO / NO-GO / STANDBY / UP / DOWN), never color alone. Numbers are the texture of the screen: latency in ms, hit / kept / discarded counts, per-stage durations, engine tallies — all set in system mono with tabular numerals, and every unknown value written as an em dash rather than a guess.

Three facts where the build diverges from the locked sketch, and the build governs: the sketch's JetBrains Mono webfont was declined — system sans for labels and prose, system mono for measurements is a user-confirmed constraint — and the sketch's inset panel shadow, glowing status dots, and 60px grid overlay did not ship; the console is strictly flat and untextured. The sketch's extra launch controls (REFINE QUERY, PAUSE) were not built either: the single launch action is SEARCH DARK WEB.

**Key Characteristics:**
- Flight-control density: telemetry is labeled, values align right, and no type exceeds the 26px brand wordmark.
- Flat, hairline, shadowless surfaces on a four-step near-black tonal ladder.
- One action blue; exactly three status hues; every state worded in a pill or chip.
- System mono with tabular numerals for every number, id, host, model name, and filename; "—" for anything unknown.
- Sharp geometry: 2px corners everywhere; circles exist only as signal dots and the slider thumb.
- Calm in long sessions — no engagement mechanics, no decorative imagery, no motion beyond data arriving.

## Colors

A cold blue-gray console palette: four near-black grounds and three text grays on a single hue family (~258°), one action blue, and exactly three status hues with matching 8% tints and tinted hairlines.

### Primary
- **Action Blue** (#4C8DFF): things you can press and the live channel. The SEARCH DARK WEB fill, the global 1px focus ring, the text-selection ground (with on-accent ink), the caret, link and underline color, the live stage cell's number and top indicator, the strip's progress rule, mission-log line ids, and the download link's hover border.

### Secondary
- **GO Green** (#3ED598): the solid GO verdict pill (on-go ink), engine UP chips, completed stage numbers, and the online tor dot.
- **Degraded Amber** (#FFB020): soft states — STANDBY, NOT CHECKED, GO · DEGRADED, halted stages — on an 8% amber tint with a tinted hairline (#4A3A14 × warn-tint).
- **Down Red** (#FF5A5A): the solid NO-GO pill (on-down ink), DOWN engine chips, TOR UNREACHABLE, and the pipeline-halted alert.

### Neutral
- **Deck Ground** (#0B0D10): the page ground behind everything.
- **Strip Band** (#0E1216): the phase strip's band, one step up from the ground.
- **Instrument Well** (#0F1319): recessed controls — inputs, selectboxes, textareas, utility buttons, inactive pills.
- **Panel Surface** (#15191F): every panel body, one step above the ground.
- **Panel Hairline** (#232830): panel and band borders. **Inner Hairline** (#1D222A): dividers between rows inside a panel. **Control Stroke** (#2A3038): widget strokes (selectbox, textarea, scrollbar thumb).
- **Console Ink** (#E8EAF0): primary text. **Control Ink** (#C9D0DC): utility-button labels. **Readout Slate** (#B8BFCC): log text and engine/source names. **Heading Slate** (#9AA2B2): panel titles. **Muted Slate** (#8C94A4): meta values and inactive text. **Faint Slate** (#7C8494): micro-labels, timestamps, empty states — the tightest text step, still clearing 4.5:1 on every surface it lands on (≈4.7:1 on panel surfaces; the WCAG-AA-minded baseline).
- **On-color inks** (#06121F on action, #06170E on GO, #1F0505 on NO-GO): the dark inks that make the solid fills legible.

### Named Rules
**The One Action Blue Rule.** The action blue is reserved for things you can press and the live channel. It is never decoration and there is never a second accent hue; status color never doubles as an accent.
**The Labeled Pill Rule.** No state is ever color alone. Every status arrives as a text-bearing pill or chip — GO, GO · DEGRADED, NO-GO, STANDBY, CHECKING, KEY SET, NOT CHECKED, READY, UP, DOWN — with color as redundant reinforcement.
**The Two Solid Pills Rule.** Only the decisive verdicts earn a solid fill: GO on green, NO-GO on red. Every softer state wears an 8% tint of its hue plus a tinted hairline.

## Typography

**Display Font:** none — the system has no display scale; the largest type on the screen is the 26px brand wordmark.
**Body Font:** system sans (`system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`)
**Label/Mono Font:** system mono (`ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace`) with `font-variant-numeric: tabular-nums`

**Character:** A two-voice console. The sans speaks — names, labels, prose, buttons; the mono measures — every number, id, host, filename, model name, and instrument key. No webfont ships (user-confirmed constraint): the type is the platform's own, and the console reads as an installed instrument rather than a styled page.

### Hierarchy
- **Brand** (700, 26px, 0.18em tracking): the ROBIN wordmark, uppercase, alone in its weight class; the tagline sits beside it at 11px, 0.22em tracked caps, faint.
- **Value** (500, 15px): board row names and the header's Elapsed / State readouts; launch-meta values sit at 14px.
- **Body** (400, 13px / 1.55): mission-log lines and alert copy; engine and source names 12.5px; empty-state copy 12.5px.
- **Label** (600, 10–12.5px, 0.18–0.22em, uppercase): widget labels, panel titles, stage names, status-block keys, footer lines. Streamlit's own widget labels are rewritten into this micro-label style (10.5px, 0.18em, faint) so native controls join the world.
- **Data** (mono, 11–12.5px, tabular): ids, counts, latencies, durations, hosts, filenames; the strip's status key at 10px; source hostnames truncate with an ellipsis rather than wrap.
- **Query** (mono, 17px): the 52px launch field and the dropdown items.

### Named Rules
**The Two Voices Rule.** System sans speaks; system mono measures. Prose never wears mono, and a figure never wears the sans — the only exception is a status string that is itself an instrument reading (PHASE 02 / 06, NOTHING LAUNCHED), which is keyed in mono.
**The Readout Rule.** Every number the tool knows is shown as its real value in tabular mono — latency in ms, counts, durations, character counts — and anything unknown is written "—", never a guess or a placeholder figure.
**The 72ch Measure Rule.** Report prose is capped at 72ch (`.st-key-report_body` markdown container); the summary reads like a findings document, never a full-width wall of text.

## Layout

The screen is a full-bleed console, not a centered page: three columns — a 336px left rail, a 660px center column, a 380px right column — inside 24px page gutters at full width (`max-width: 100%`), with a 14px vertical rhythm between elements. Above them, a 72px header (brand lockup left; tor chip, Elapsed, State right) and the phase strip: six flex cells of at least 118px with 16px inner padding, closed by a 230px status block carrying the mode key and a 3px progress rule.

Every panel spans its column, and the column stacks are the workflow order. Left rail: Config → Research domains → Prompt → Past investigations. Center: Pre-flight · GO / NO-GO → Launch → Report · Findings. Right: Mission log → Sources → Engines. Row densities are the table rhythm: 50px checklist rows, 32px source rows, 30px engine rows, 9px/16px log rows.

Under 760px the console collapses to one column: 14px gutters, strip cells three per row, checklist rows stack name-over-pill, source hostnames drop out, engine rows tighten to 58px columns, and the whole stack reads in the same order.

### Named Rules
**The Preflight-First Rule.** The center column reads board → launch → report, and that order is the contract: the GO/NO-GO board is always the first thing read, above the launch field, in every state.

## Elevation & Depth

Flat by construction: no box-shadow, no gradient, no blur, no backdrop-filter anywhere on the screen. Depth comes from a four-step tonal ladder (Deck Ground #0B0D10 → Strip Band #0E1216 → Instrument Well #0F1319 → Panel Surface #15191F) read through 1px hairlines, and state is tinted, not raised. The hover vocabulary is a line swap — borders and text shift toward the action blue or brighter ink. The primary launch button is the only large filled surface on the screen; the status pills are its miniature.

### Shadow Vocabulary
None. The system declines shadows entirely; a shadow in this world reads as a defect, the way a lifted panel would in a real cockpit.

### Named Rules
**The Flat Console Rule.** Edges are hairlines and depth is tonal. The sketch's inset panel shadow and glowing status dots did not ship; do not reintroduce lifts, glows, or gradients.
**The State-Is-Tint Rule.** Raised state is a translucent hue wash — the live stage at rgba(76,141,255,.10), the halted stage at rgba(255,176,32,.08) — never a lift, a heavier border, or a glow.

## Shapes

Rectangular at 2px, everywhere. Panels, pills, chips, buttons, inputs, selectboxes, and textareas all share the single corner radius (2px); the world never fully rounds a control, so the status "pills" are rectangle-labels, not capsules. Circles appear only as physical signals: the 8px tor dot, the circular slider thumb (ink fill with a 3px action-blue ring), and the 6px-radius scrollbar thumb. The one rule heavier than 1px is the phase strip's 2px colored top edge on the live or halted stage cell — a channel indicator, not a decorative stripe.

### Named Rules
**The 2px Corner Rule.** One radius for the whole console. Nothing gets rounder with size, importance, or emphasis.
**The Signal-Circle Rule.** Roundness is reserved for physical signals — dots and thumbs. A rounded container is a shape this world does not have.

## Components

### Buttons
- **Shape:** 2px corners, 1px hairline border, uppercase tracked labels at 7–800 weight.
- **Primary (SEARCH DARK WEB):** full-width, 54px, action-blue fill with on-accent ink, 14px/800/0.14em caps; hover deepens to #3D7FF0 (#4C8DFF → #3D7FF0), instant swap with no transition declared; the only large filled surface on the screen.
- **Utility (Re-run pre-flight, Load record, Clear record):** 40px, instrument-well fill, control-ink label at 11.5px/700/0.12em; hover swaps the border to action blue and the label to ink. The pre-flight re-run is a quieter 30px variant with transparent fill and faint label.
- **Focus:** every control takes the global ring — `1px solid` action blue, offset 2px.

### Chips
- **Status pills:** 11px/700/0.14em caps, 5px 10px, 2px corners. Solid for GO / NO-GO; 8% hue tint plus tinted hairline for STANDBY, CHECKING, KEY SET, NOT CHECKED, GO · DEGRADED.
- **Engine chips:** 10px/700/0.12em caps, 4px 8px — UP in go-tint with a go-line border, DOWN in down-tint with a down-line border.
- **Tor chip:** the header's state light — dot plus worded state (TOR CONNECTED / TOR UNREACHABLE / TOR · CHECKING…), 12px/600, 0.08em tracking, 7px 12px, with its own tinted fill and hairline for on / off / idle.

### Cards / Containers
- **Corner Style:** 2px.
- **Background:** Panel Surface over Deck Ground; recessed controls use Instrument Well.
- **Shadow Strategy:** none — see Elevation & Depth.
- **Border:** 1px Panel Hairline around each panel; Inner Hairline between rows, with the last row's divider dropped.
- **Headers:** 38px strips — tracked-caps title left (11px/600/0.2em, Heading Slate), mono meta right (10px, faint) — always present, even on tables.
- **Internal Padding:** 12px 14px 14px for panel bodies; rows carry their own 10px 16px.

### Inputs / Fields
- **Query field:** the launch instrument — Instrument Well, a 1px control-blue-gray stroke (#2E3A4C), 2px corners, 52px tall, 17px mono ink, faint mono placeholder.
- **Selectboxes / textareas / popovers:** Instrument Well at 42px, mono content (textarea 12.5px), popover items mono.
- **Sliders:** 6px track, action-blue fill, circular ink thumb with 3px blue ring, mono thumb value; all three sliders (threads, max results, max pages) follow this one skin.
- **Disabled / Empty:** a disabled textarea keeps the well but mutes to dim ink on a darker ground with a not-allowed cursor — visible as off, never invisible.
- **Labels:** all native widget labels are converted to the 10.5px/600/0.18em faint micro-label; help affordances keep Streamlit's position but drop to faint.

### Navigation
There is no navigation chrome. The fixed furniture is the 72px header — brand lockup left, tor chip plus Elapsed / State readouts right — and the footer strip: a status line left, the author credit centered (a durable product commitment), and a mono status string right. Streamlit's own header, toolbar, menu, decoration, and sidebar are hidden outright; the tool is a single screen with a single launch action, and the archive rail (Past investigations, Load record) is the only way backward.

### Signature Components
- **Pre-flight · GO / NO-GO board.** The world's front door: a fixed-order checklist — Tor network, selected provider, unconfigured keys, local models, engines n / 16 — each row a 15px name, an 11px faint meta line, and a verdict pill. It reads before the launch field and re-runs on demand (RE-RUN PRE-FLIGHT). Its row order is part of the contract; new checks append, they do not reorder.
- **Phase strip.** Six stage cells (01 LOAD · LLM → 06 SUMMARY) whose number, name, and duration are all real: a completed cell shows its measured ms, the live cell shows RUNNING over a blue wash and 2px top indicator, a halted cell shows HALT in amber. The status block keys the mode (READY · PRE-FLIGHT, PHASE 03 / 06, MISSION COMPLETE, MISSION HALTED, ARCHIVE · timestamp) with a 3px progress rule that advances only on completed stages.
- **Mission log.** Numbered mono lines (01, 02, …) recording real counts as they happen — hits, kept, discarded, pages, the saved filename — the run's black box.
- **Run-meta readout.** A five-cell instrument row under the launch button: Phase, Threads, Pages, Filtered, Hits discarded — values mono, updated per stage, "—" before a run.
- **Error alert.** A down-tinted panel titled PIPELINE HALTED · FAILED TO <STAGE> with the raw error in mono and recovery hints for the specific provider; the footer then carries HALTED · <STAGE>.
- **Empty states.** Every empty panel says what will fill it and how ("No entries yet — the log fills when a mission launches"), in 12.5px faint — never a blank box.

### Named Rules
**The Measured-Strip Rule.** A stage cell shows a duration only when that stage has actually completed with a measured millisecond count; before that it shows nothing, RUNNING, or HALT. No invented or estimated durations, ever.

## Do's and Don'ts

### Do:
- **Do** read state before action: keep the GO/NO-GO board first in the center column, and keep its fixed row order (Tor → selected provider → unconfigured keys → local models → engines) wherever a pre-flight check is drawn.
- **Do** carry every status as a text-bearing pill or chip; color is redundant reinforcement, never the message.
- **Do** set every number, id, host, filename, and duration in system mono with tabular numerals, and write "—" when a value is unknown.
- **Do** keep corners at 2px and edges at 1px hairlines; circles only for the tor dot and slider thumb.
- **Do** reserve action blue for pressable things and the live channel, and hold the palette to exactly three status hues.
- **Do** keep tracked caps (0.18em–0.22em) at 10–12.5px and reserve them for labels, statuses, and buttons — every caps label here names an instrument or a state.
- **Do** keep report prose at 72ch and the author credit in the footer; both are confirmed commitments.
- **Do** ship system sans and system mono only — no webfonts (user-confirmed constraint).

### Don't:
- **Don't** add shadows, glows, gradients, blur, or background grid overlays; flatness is the world, and the sketch's inset/glow effects deliberately did not ship.
- **Don't** add webfonts, icon fonts, or glyph/emoji icons; the screen ships no raster imagery at all (the logo lives only as the page favicon).
- **Don't** fill a soft state with a solid status color — solids belong to GO and NO-GO alone.
- **Don't** use status hues as decoration or as a second accent; that role belongs to action blue only.
- **Don't** set prose, log text, or report copy in mono, and don't set paragraphs in tracked caps — caps belong to labels, statuses, and buttons.
- **Don't** round a container past 2px or introduce capsule-shaped controls; status "pills" are rectangles with 2px corners.
- **Don't** show a stage duration before the stage has measured one, and don't animate with authored transitions — state swaps are instant; the only movement is data arriving (progress rule, streaming report, spinners).
- **Don't** reintroduce Streamlit's own chrome (header, toolbar, sidebar, menu, decoration) or any second launch path; SEARCH DARK WEB is the one primary action.
