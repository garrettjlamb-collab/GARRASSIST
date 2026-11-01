# Court review: the coach approves a drawn court instead of marking points

**Why.** Reference pictures cut the job from 40 frames to 3-5, but each picture still walks Garrett through up to 22
named points (near, far, left, right). He does not want to do it. He asked for the machine to mark and for him to review.

**Evidence (2026-10-31).** Claude marked queued pictures by eye on a gridded crop. On the Prescott frame Garrett had
marked, with 9 marks made without seeing his, the two courts were a median 1 ft apart and 3 ft at worst over the half
court (84 ft court). On Mesquite, 6 marks at one end lined up the lane, arc and top sideline but missed the centre line,
and one extra midcourt mark made the near sideline about 100 px worse. Good enough to review; not good enough to trust.

## The flow

1. **A first draft.** In a session, Claude views each queued picture through `film court-draft view`, picks the points
   and saves them with `film court-draft set`, which fits them, reports any it throws out and writes a picture of the
   result to check. This spends Claude usage, by Garrett's choice, until a model can draw them.
2. **Review.** The Label court tab shows a picture holding a draft with the whole court drawn on it. `Y` (or `S`, or
   Enter) approves. If the lines are off he grabs the court anywhere on the floor and pulls that spot onto the paint; the
   court redraws as he drags. `L` steps the court length (74, 84, 94 ft), `H` hides the lines, `Backspace` lets go of the
   last spot he pulled, `R` returns to the draft, `Z` zooms, `D` skips the picture.
3. **Carrying.** An approved picture becomes a reference. The game's other queued pictures that line up with the
   references get their drafts redrawn from them, so later pictures start closer.
4. **Afterwards nothing changes.** `film court`, `--refs-from` and `court.parquet` read references as they do today.

A picture with no draft keeps the old point-by-point walk, so nothing that exists today breaks.

## Decisions

- **Pulled spots outweigh draft marks.** The fit is least squares over every mark, with each pulled spot counted 100
  times, so what he pulled stays under the mouse and a wrong draft mark loses without having to be found and deleted.
- **Pulled spots can be anywhere on the floor**, not only at the 22 points, because a line is what he sees. They are
  sent as feet, read through the current fit at the moment he grabs.
- **Approving saves every template point in view through the approved fit**, not the marks. Both ends and midcourt then
  sit in one reference when the picture shows them, so `measure_length` recovers the length he approved and no label
  format changes. Saved labels stay the 22 named points.
- **A mirrored fit is refused while dragging.** The last good court stays drawn and a note says so. Approve is refused
  too. The orientation guard is the one `fit_mapping` already uses.
- **Carried marks just outside the frame stay in the draft.** They cannot be saved as labels, but they hold the fit
  across the court; a Prescott picture with 4 carried marks in frame gave no court until the other 7 were kept.
- **A carried draft replaces a Claude draft only when it holds 6 or more points.** A carried fit from a few clustered
  points can be worse than a draft that spans the picture. A picture with no draft takes any carried fit.
- **Every approval is logged** to `labels/court/review_log.jsonl`: who drew the draft, how many spots he pulled, how far
  the court moved at those spots, and the seconds spent. That measures the "about a minute a game" guess and tells us
  when a model's drafts are good enough to take over.

## Units

| Unit | Does |
|---|---|
| `film/interpret/court.py: fit_handles` | The weighted fit, sharing the orientation guard with `fit_mapping` |
| `film/export/court_sheet.py: court_polylines` | The court's lines in pixels, used by `draw_court` and the page |
| `film/perceive/court_draft.py` | Draw for the page, approve, save a draft, the gridded view, carry drafts, the log |
| `film/commands/court_draft.py` | `film court-draft list / view / set` |
| `film/review/app.py` | `POST /api/label/court/{id}/draw` and `/approve` |
| `film/review/static/index.html` | The review mode of the Label court tab |

## Out of scope

- **The trained model that takes over drafting.** It learns from approved pictures through `film court --train-frames`
  and `film train court`. It comes once several gyms are approved.
- **Snapping a draft onto the painted lines** by image processing. Arcadia's floor has other sports' lines. Worth testing
  if the log shows drafts need a lot of pulling.
- **A program that finds the court with no starting point.** Garrett chose drafts by Claude first.
- **The shot dots on the cards.** Those are plan B of shot locations.

## Testing

Unit tests on the weighted fit (a pulled spot stays where it was put; a mirror is refused), the polylines, the draft
save (bad marks reported, mirror refused), carrying (replaces only on the rule above) and the API (draw, approve saves
points through the fit, the log line, a refused approve). The page is driven in a browser on the real queue before
Garrett sees it.
