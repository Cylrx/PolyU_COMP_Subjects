/*
 * Copyright (c) 2024 Mehdi Essalehi & Amine Hadnane
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. 
 */

/*
 * Copyright (c) 2025 Wang Yuqi
 * Modifications to original work by Mehdi Essalehi & Amine Hadnane.
 * Original work: https://github.com/essmehdi/ensias-report-template?tab=MIT-1-ov-file
 */

#let IMAGE_BOX_MAX_WIDTH = 120pt
#let IMAGE_BOX_MAX_HEIGHT = 50pt

#let project(
  title: "", 
  subtitle: "Thesis", 
  school-logo: none, 
  company-logo: none, 
  authors: (), 
  mentors: (), 
  jury: (), 
  branch: none, 
  academic-year: none, 
  footer-text: "COMPXXXX", 
  body
) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)

  set page(
  numbering: "1",
  number-align: center,
  footer: context {
    // Omit page number on the first page
    let page-number = counter(page).at(here()).at(0)
    if page-number > 1 {
      line(length: 100%, stroke: 0.5pt)
      v(-2pt)
      text(size: 12pt, weight: "regular")[
        #footer-text
        #h(1fr)
        #page-number
        #h(1fr)
        #academic-year
      ]
    }
  }
)

  let dict = json("./resources/i18n/en.json")
  let lang = "en"

  set text(font: "Libertinus Serif", lang: lang, size: 13pt)
  set heading(numbering: "1.1")
  
  show heading: it => {
    if it.level == 1 and it.numbering != none {
      pagebreak()
      v(10pt)
      text(size: 30pt)[#it.body ]
      v(10pt)
    } else {
      v(5pt)
      [#it]
      v(12pt)
    }
  }

  block[
    #box(height: IMAGE_BOX_MAX_HEIGHT, width: IMAGE_BOX_MAX_WIDTH)[
      #align(left + horizon)[
        #company-logo
      ]
    ]
    #h(1fr)
    #box(height: IMAGE_BOX_MAX_HEIGHT, width: IMAGE_BOX_MAX_WIDTH)[
      #align(right + horizon)[
        #if school-logo == none {
          image("images/ENSIAS.svg")
        } else {
          school-logo
        }
      ]
    ]
  ]
  
  // Title box  
  align(center + horizon)[
    #if subtitle != none {
      text(size: 14pt, tracking: 2pt)[
        #smallcaps[
          #subtitle
        ]
      ]
    }
    #line(length: 100%, stroke: 0.5pt)
    #text(size: 20pt, weight: "bold")[#title]
    #line(length: 100%, stroke: 0.5pt)
  ]

  // Credits
  box()
  h(1fr)
  grid(
    columns: (auto, 1fr, auto),
    [
      // Authors
      #if authors.len() > 0 {
        [
          #text(weight: "bold")[
            #if authors.len() > 1 {
              dict.author_plural
            } else {
              dict.author
            }
            #linebreak()
          ]
          #for author in authors {
            [#author #linebreak()]
          }
        ]
      }
    ],
    [
      // Mentor
      #if mentors != none and mentors.len() > 0 {
        align(right)[
          #text(weight: "bold")[
            #if mentors.len() > 1 {
              dict.mentor_plural
            } else {
              dict.mentor
            }
            #linebreak()
          ]
          #for mentor in mentors {
            mentor
            linebreak()
          }
        ]
      }
      // Jury
      #if jury != none and jury.len() > 0 {
        align(right)[
          *#dict.jury* #linebreak()
          #for prof in jury {
            [#prof #linebreak()]
          }
        ]
      }
    ]
  )

  align(center + bottom)[
    #if branch != none {
      branch
      linebreak()
    }
    #if academic-year != none {
      [#dict.academic_year: #academic-year]
    }
  ]
  

  body
}
