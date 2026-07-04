#figure(
  caption: [Dispatch results across three arrival patterns at mean arrival rate 80. FxE: FixedEdge, FxO: FixedOffload (reference baselines, always use full model); AdQ: AdaptQueue, AdD: AdaptDeadline, 1S: One-Step Optimal, MS: Multi-Step Optimal. Among AdQ/AdD/1S/MS, *bold* marks best and #underline[underline] second-best. DR: throughput ($s^(-1)$); Lat., AoI, P95, P99: ms.],
  {set text(size: 9.5pt)
  show table.cell.where(y: 1): it => {
    set text(size: 0.85em)
    pad(top: 3pt, bottom: 3pt)[#it]
  }
  show table.cell.where(y: 2): it => pad(top: 2pt)[#it]
  show table.cell.where(y: 8): it => pad(bottom: 2pt)[#it]
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    inset: (x: 2.5pt, y: 2.5pt),
    column-gutter: (1pt, -3pt, -3pt, -3pt, -3pt, -3pt, 6pt, -3pt, -3pt, -3pt, -3pt, -3pt, 6pt, -3pt, -3pt, -3pt, -3pt, -3pt),
    row-gutter: (0pt, 0pt, 0pt, 0pt, 0pt, 0pt, 0pt, 0pt),
    toprule,
    table.header(
      [], table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*Uniform*],
          table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*Poisson*],
          table.cell(colspan: 6, align: center, inset: (top: 5pt, bottom: 5pt))[*Gamma ($alpha=0.05$)*],
      table.hline(start: 1, end: 7, stroke: 0.05em),
      table.hline(start: 7, end: 13, stroke: 0.05em),
      table.hline(start: 13, end: 19, stroke: 0.05em),
      [Metrics], [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
                  [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
                  [FxE], [FxO], [AdQ], [AdD], [*1SO*], [*MSO*],
    ),
    midrule,
    [DR#sym.arrow.t],   [80.4], [80.4], [23.0], [75.7], [*80.4*], [#underline[80.4]],
              [80.5], [80.5], [21.2], [79.7], [*80.5*], [#underline[80.5]],
              [81.7], [81.7], [23.2], [36.8], [*82.1*], [#underline[81.5]],
    [Acc.#sym.arrow.t], [99.0], [99.0], [*99.2*], [99.1], [#underline[99.2]], [99.2],
              [99.0], [99.0], [99.1], [99.2], [*99.2*], [#underline[99.2]],
              [99.0], [99.0], [99.1], [*99.2*], [#underline[99.2]], [99.2],
    [Lat.#sym.arrow.b], [4], [5], [408], [7], [*4*], [#underline[4]],
              [4], [6], [413], [*5*], [5], [#underline[5]],
              [6], [12], [235], [112], [#underline[11]], [*9*],
    [AoI#sym.arrow.b],  [9], [11], [450], [33], [*10*], [#underline[10]],
              [13], [15], [453], [16], [*14*], [#underline[14]],
              [119], [120], [349], [225], [*119*], [#underline[119]],
    [P95#sym.arrow.b],  [1], [1], [592], [12], [*1*], [#underline[1]],
              [1], [1], [591], [*1*], [1], [#underline[1]],
              [7], [92], [532], [317], [#underline[86]], [*74*],
    [P99#sym.arrow.b],  [158], [156], [598], [241], [*123*], [#underline[131]],
              [182], [165], [598], [*122*], [137], [#underline[135]],
              [159], [193], [591], [360], [#underline[173]], [*138*],
    [%Miss#sym.arrow.b], [0.0], [0.0], [88.5], [#underline[0.4]], [*0.0*], [*0.0*],
              [0.0], [0.0], [90.6], [0.1], [#underline[0.0]], [*0.0*],
              [0.0], [0.0], [74.4], [1.7], [#underline[0.0]], [*0.0*],
    botrule,
  )},
) <tab:results>
