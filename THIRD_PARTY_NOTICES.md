# Third-party content

The MIT licence in `LICENSE` covers this repository's own code. It does **not**
cover the third-party font and icon files vendored under `plotting/`, which
remain under their own licences and are redistributed here so that figures
render identically without a system font install.

## Fonts (`plotting/fonts/`)

| family | files | upstream | licence |
| --- | --- | --- | --- |
| TeX Gyre Pagella | `texgyrepagella-{regular,bold}.otf` | GUST e-foundry | GUST Font License, see `plotting/fonts/GUST-FONT-LICENSE.txt` |

The GFL text ships alongside the fonts, as that licence requires.

Computer Modern math fonts are supplied by Matplotlib; no copies are vendored.

## Icons (`plotting/assets/`)

### Font Awesome Free

`fontawesome/svgs/solid/{hand,key,minus,pause}.svg`.

- **Creator:** Font Awesome, by Fonticons, Inc.
- **Source:** https://fontawesome.com --- release 7.2.0, from
  `https://github.com/FortAwesome/Font-Awesome.git` at commit `337dd2045`
- **Licence:** the icons are Creative Commons Attribution 4.0 International
  (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, within the Font
  Awesome Free licence, https://fontawesome.com/license/free. Non-icon files
  such as the metadata are MIT under the same licence. The upstream text is in
  `fontawesome/LICENSE.txt`.
- **Changes made:** the `.svg` files are unmodified and keep their upstream
  attribution comments. The icons are geometrically transformed when converted
  into plot markers.
