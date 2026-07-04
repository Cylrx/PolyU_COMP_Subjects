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
    [DR#sym.arrow.t],   [38.4], [48.0], [35.5], [12.3], [#underline[38.5]], [*55.7*],
              [38.2], [47.9], [23.1], [30.9], [#underline[39.2]], [*64.0*],
              [33.4], [40.9], [15.8], [19.9], [#underline[38.6]], [*42.3*],
    [Acc.#sym.arrow.t], [89.0], [89.1], [87.3], [*89.3*], [#underline[88.5]], [88.3],
              [89.2], [89.1], [87.7], [#underline[88.3]], [*89.3*], [87.7],
              [88.5], [88.9], [82.9], [87.6], [*89.0*], [#underline[88.2]],
    [Lat.#sym.arrow.b], [367], [380], [395], [#underline[300]], [341], [*252*],
              [361], [372], [420], [#underline[263]], [333], [*220*],
              [248], [251], [344], [245], [#underline[224]], [*176*],
    [AoI#sym.arrow.b],  [381], [377], [423], [#underline[337]], [353], [*252*],
              [374], [369], [452], [#underline[275]], [345], [*218*],
              [277], [265], [388], [295], [#underline[252]], [*211*],
    [P95#sym.arrow.b],  [378], [462], [556], [385], [#underline[351]], [*321*],
              [377], [457], [612], [365], [#underline[351]], [*305*],
              [370], [398], [584], [348], [#underline[345]], [*298*],
    [P99#sym.arrow.b],  [381], [490], [617], [403], [#underline[354]], [*354*],
              [381], [486], [621], [390], [#underline[355]], [*343*],
              [377], [460], [615], [375], [#underline[359]], [*325*],
    [%Miss#sym.arrow.b], [99.7], [97.8], [96.8], [29.8], [#underline[6.9]], [*1.4*],
              [91.4], [86.9], [95.7], [9.7], [#underline[6.4]], [*0.7*],
              [59.0], [53.0], [87.1], [4.7], [#underline[2.2]], [*0.2*],
    botrule,
  )},
) <tab:results>
