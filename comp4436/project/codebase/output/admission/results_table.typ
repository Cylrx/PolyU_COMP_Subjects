#figure(
  caption: [Admission control ablation under Gamma arrival (rate = 80). FxE: FixedEdge, FxO: FixedOffload; AdQ: AdaptQueue, AdD: AdaptDeadline, 1SO: One-Step Optimal, MSO: Multi-Step Optimal. *Bold* marks the better admission policy per dispatcher. DR: throughput ($s^(-1)$); Acc.: %; Lat.: ms.],
  {set text(size: 9.5pt)
  show table.cell.where(y: 1): it => {
    set text(size: 0.85em)
    pad(top: 3pt, bottom: 3pt)[#it]
  }
  show table.cell.where(y: 2): it => pad(top: 2pt)[#it]
  show table.cell.where(y: 3): it => pad(bottom: 2pt)[#it]
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    inset: (x: 2.5pt, y: 2.5pt),
    column-gutter: (1pt, -3pt, -3pt, -3pt, -3pt, -3pt, 6pt, -3pt, -3pt, -3pt, -3pt, -3pt, 6pt, -3pt, -3pt, -3pt, -3pt, -3pt),
    row-gutter: (0pt, 0pt, 0pt),
    toprule,
    table.header(
      [], table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*DR#sym.arrow.t*],
          table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*Acc.#sym.arrow.t*],
          table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*Lat.#sym.arrow.b*],
      table.hline(start: 1, end: 7, stroke: 0.05em),
      table.hline(start: 7, end: 13, stroke: 0.05em),
      table.hline(start: 13, end: 19, stroke: 0.05em),
      [Policy], [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
                  [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
                  [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
    ),
    midrule,
    [DrO],              [81.7], [81.7], [*23.2*], [36.8], [82.1], [*81.5*],
              [99.0], [99.0], [99.1], [*99.2*], [99.2], [*99.2*],
              [*6*], [*12*], [*235*], [112], [*11*], [*9*],
    [DrT],              [81.7], [81.7], [22.4], [*39.4*], [82.1], [81.4],
              [99.0], [99.0], [*99.2*], [99.1], [99.2], [99.2],
              [8], [13], [236], [*109*], [11], [9],
    botrule,
  )},
) <tab:admission>
