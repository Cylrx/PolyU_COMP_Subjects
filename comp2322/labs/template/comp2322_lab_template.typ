/*
 * Copyright (c) 2025 Wang Yuqi
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

#import "lib.typ": project
#import "@preview/wrap-indent:0.1.0": wrap-in, allow-wrapping

#set raw(tab-size: 4)

#show terms.item: allow-wrapping
#show link: set text(fill: blue)
#show link: underline

#show "next_page": [
  #align(center)[
    #v(3em)
    #text(size: 16pt, weight: "black")[
      (See Next Page)
    ]
  ]
]

#let answer-block(content) = {
  text(
    [
      #rect(
        [
          #text(size: 1.15em)[*Answer: *]
          #content
        ],
        fill: gray.lighten(95%),
        width: 100%,
        stroke: gray.lighten(40%),
        inset: 1em
      )
    ]
  )
}

#show: project.with(
  title: "Lab X Report: Something",
  subtitle: "COMP2322 Computer Networking",
  authors: (
    "Name\nStudent ID",
  ),
  mentors: (
    "Dr. LOU Wei",
  ),
  school-logo: image("polyu-logo.png"), // top right
  branch: "BsC in Computer Science",
  academic-year: "2024/2025 Sem 2",
  footer-text: "COMP2322" // Text used in left side of the footer
)

= Questions