<!DOCTYPE html>

<html>
<head><meta charset="utf-8"/>
<title>Bezier</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>
<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>
<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link href="custom.css" rel="stylesheet"/>
<!-- Loading mathjax macro -->
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
<!-- End of mathjax configuration --></head>
<body>
<div class="border-box-sizing" id="notebook" tabindex="-1">
<div class="container" id="notebook-container">
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Bezier-Curves">Bezier Curves<a class="anchor-link" href="#Bezier-Curves">¶</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">scipy.special</span> <span class="k">import</span> <span class="n">comb</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Use this to click points for Bezier curve</span>
<span class="c1"># Might have to run this block twice. (?)</span>
<span class="o">%</span><span class="k">matplotlib</span> osx
<span class="n">plt</span><span class="o">.</span><span class="n">ion</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Bernstein-polynomials">Bernstein polynomials<a class="anchor-link" href="#Bernstein-polynomials">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">B</span><span class="p">(</span><span class="n">i</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">t</span><span class="p">):</span>
    <span class="n">val</span> <span class="o">=</span> <span class="n">comb</span><span class="p">(</span><span class="n">N</span><span class="p">,</span><span class="n">i</span><span class="p">)</span> <span class="o">*</span> <span class="n">t</span><span class="o">**</span><span class="n">i</span> <span class="o">*</span> <span class="p">(</span><span class="mf">1.</span><span class="o">-</span><span class="n">t</span><span class="p">)</span><span class="o">**</span><span class="p">(</span><span class="n">N</span><span class="o">-</span><span class="n">i</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">val</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Plot-the-Bernstein-polynomials">Plot the Bernstein polynomials<a class="anchor-link" href="#Plot-the-Bernstein-polynomials">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">tt</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">N</span> <span class="o">=</span> <span class="mi">7</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="o">+</span><span class="mi">1</span><span class="p">):</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">tt</span><span class="p">,</span> <span class="n">B</span><span class="p">(</span><span class="n">i</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">tt</span><span class="p">));</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Bezier-Curves">Bezier Curves<a class="anchor-link" href="#Bezier-Curves">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">P</span><span class="p">(</span><span class="n">t</span><span class="p">,</span> <span class="n">X</span><span class="p">):</span>
    <span class="sd">'''</span>
<span class="sd">     xx = P(t, X)</span>
<span class="sd">     </span>
<span class="sd">     Evaluates a Bezier curve for the points in X.</span>
<span class="sd">     </span>
<span class="sd">     Inputs:</span>
<span class="sd">      X is a list (or array) or 2D coords</span>
<span class="sd">      t is a number (or list of numbers) in [0,1] where you want to</span>
<span class="sd">        evaluate the Bezier curve</span>
<span class="sd">      </span>
<span class="sd">     Output:</span>
<span class="sd">      xx is the set of 2D points along the Bezier curve</span>
<span class="sd">    '''</span>
    <span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
    <span class="n">N</span><span class="p">,</span><span class="n">d</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">shape</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>   <span class="c1"># Number of points, Dimension of points</span>
    <span class="n">N</span> <span class="o">=</span> <span class="n">N</span> <span class="o">-</span> <span class="mi">1</span>
    <span class="n">xx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">t</span><span class="p">),</span> <span class="n">d</span><span class="p">))</span>
    
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="o">+</span><span class="mi">1</span><span class="p">):</span>
        <span class="n">xx</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">outer</span><span class="p">(</span><span class="n">B</span><span class="p">(</span><span class="n">i</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">t</span><span class="p">),</span> <span class="n">X</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    
    <span class="k">return</span> <span class="n">xx</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">[</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">clf</span><span class="p">()</span>

<span class="n">clickable</span> <span class="o">=</span> <span class="kc">False</span>

<span class="k">if</span> <span class="n">clickable</span><span class="p">:</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span><span class="s1">'w.'</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">'equal'</span><span class="p">);</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">draw</span><span class="p">()</span>
    <span class="n">c</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ginput</span><span class="p">(</span><span class="mi">20</span><span class="p">,</span> <span class="n">mouse_stop</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span> <span class="c1"># on macbook, alt-click to stop</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">draw</span><span class="p">()</span>
<span class="k">else</span><span class="p">:</span>
    <span class="n">c</span> <span class="o">=</span> <span class="p">[(</span><span class="mf">0.09374999999999989</span><span class="p">,</span> <span class="mf">0.15297619047619054</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.549107142857143</span><span class="p">,</span> <span class="mf">0.1648809523809524</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.7083333333333335</span><span class="p">,</span> <span class="mf">0.6142857142857144</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.5282738095238095</span><span class="p">,</span> <span class="mf">0.8940476190476193</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.24404761904761907</span><span class="p">,</span> <span class="mf">0.8776785714285716</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.15327380952380942</span><span class="p">,</span> <span class="mf">0.6321428571428573</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.580357142857143</span><span class="p">,</span> <span class="mf">0.08303571428571432</span><span class="p">),</span>
         <span class="p">(</span><span class="mf">0.8839285714285716</span><span class="p">,</span> <span class="mf">0.28988095238095246</span><span class="p">)]</span>
    
<span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">c</span><span class="p">)</span>

<span class="n">tt</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">200</span><span class="p">)</span>
<span class="n">xx</span> <span class="o">=</span> <span class="n">P</span><span class="p">(</span><span class="n">tt</span><span class="p">,</span> <span class="n">X</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">xx</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span> <span class="n">xx</span><span class="p">[:,</span><span class="mi">1</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">X</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span> <span class="n">X</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="s1">'ro'</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAecAAAHVCAYAAADLvzPyAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xl8VOWh//HPM9n3fYGs7PseFnGvVEUtatUKxa1Vaa+17W3totfW362Wtmqtem9tlaq1rbhXrVUUq3XDDcK+BggEEsISErJAyDbz/P5I4EaMMEAy50zm+369eE3mzMmc74jJl+cszzHWWkRERMQ9PE4HEBERkc9SOYuIiLiMyllERMRlVM4iIiIuo3IWERFxGZWziIiIy6icRUREXEblLCIi4jIqZxEREZcJd2rD6enptrCw0KnNi4iIBNTSpUv3Wmsz/FnXsXIuLCykuLjYqc2LiIgElDFmm7/rare2iIiIy6icRUREXEblLCIi4jIqZxEREZdROYuIiLiMyllERMRlVM4iIiIuo3IWERFxGZWziIiIy6icRUREXEblLCIi4jIqZxEREZfxq5yNMecbY0qMMZuNMbd28XqBMeZtY8wqY8y7xpjc7o8qIiISGo5ZzsaYMOAhYDowHJhljBl+xGq/Bf5qrR0N3An8uruDioiIhAp/Rs6TgM3W2i3W2hbgGeDiI9YZDrzd8fU7XbwuIiIifvKnnHOA8k7PKzqWdbYSuKzj60uBBGNM2pFvZIyZY4wpNsYUV1VVnUheEXGT+fOhsBA8nvbH+fOdTiTSK/hTzqaLZfaI5z8CzjTGLAfOBHYAbZ/7JmvnWWuLrLVFGRkZxx1WRFxk/nyYMwe2bQNr2x/nzFFBi3QDf8q5Asjr9DwXqOy8grW20lr7VWvtOOD2jmV13ZZSRNzn9tuhsfGzyxob25eLyEnxp5yXAIOMMf2MMZHATOCVzisYY9KNMYfe6zbg8e6NKSKus3378S0XEb8ds5yttW3AzcBCYD3wnLV2rTHmTmPMjI7VzgJKjDEbgSxgbg/lFRG3yM8/vuUi4rdwf1ay1i4AFhyx7I5OX78AvNC90UTE1ebObT/G3HnXdmxs+3IROSmaIUxETszs2TBvHhQUgDHtj/PmtS8XkZPi18hZRKRLs2erjEV6gEbOoUrXp4qIuJZGzqHo0PWph44VHro+FTQKEhFxAY2cQ5GuTxURcTWVcyjS9akiIq6mcg5Fuj5VRMTVVM6haO7c9utRO9P1qSIirqFyDkW6PlVExNV0tnao0vWpIiKupZGziIiIy6icRUREXEblLCIi4jIqZxEREZdROYuIiLiMyllERMRlVM4iIiIuo3IWERFxGZWziIiIy6icRUREXEblLCIi4jIqZxEREZdROYuIiLiMyllERMRlVM4iIiIuo3IWERFxGZWziIiIy6icRUREXEblLCIi4jIqZxEREZdROYuIiLiMyllERMRlVM4iIiIuo3IWERFxGZWziIiIy6icRUREXEblLCIi4jIqZxEREZdROYuIiLiMX+VsjDnfGFNijNlsjLm1i9fzjTHvGGOWG2NWGWMu6P6oIiIioeGY5WyMCQMeAqYDw4FZxpjhR6z2M+A5a+04YCbwh+4OKiIiEir8GTlPAjZba7dYa1uAZ4CLj1jHAokdXycBld0XUUREJLSE+7FODlDe6XkFMPmIdf4beNMY810gDpjWLelERERCkD8jZ9PFMnvE81nAE9baXOAC4G/GmM+9tzFmjjGm2BhTXFVVdfxpRUREQoA/5VwB5HV6nsvnd1tfDzwHYK39GIgG0o98I2vtPGttkbW2KCMj48QSi4iI9HL+lPMSYJAxpp8xJpL2E75eOWKd7cA5AMaYYbSXs4bGIiIiJ+CY5WytbQNuBhYC62k/K3utMeZOY8yMjtVuAW40xqwEngaus9YeuetbRERE/ODPCWFYaxcAC45Ydkenr9cBp3ZvNBERkdCkGcJERERcRuUsIiLiMipnERERl1E5i4iIuIzKWURExGVUziIiIi6jchYREXEZlbOIiIjLqJxFRERcRuUsIiLiMipnERERl1E5i4iIuIzKWURExGVUziIiIi6jchYREXEZlbOIiIjLqJxFRERcRuUsIiLiMipnERERl1E5i4iIuIzKWURExGVUziIiIi6jchYREXEZlbOIiIjLqJxFRERcRuUsIiLiMipnERERl1E5i4iIuIzKWURExGVUziIiIi6jchYREXEZlbOIiIjLqJxFRERcRuUsIiLiMipnERERl1E5i4iIuIzKWURExGVUziIiIi6jchYREXGZcKcDiPRWXp+lsvYg22saqTnQcnh5VLiHnJQY8lJjSYyOcDChiLiVX+VsjDkfeBAIAx61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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="TrueType-Fonts">TrueType Fonts<a class="anchor-link" href="#TrueType-Fonts">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">DrawBezier</span><span class="p">(</span><span class="n">p</span><span class="p">,</span> <span class="n">n</span><span class="p">):</span>

    <span class="n">x1</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">y1</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">x2</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span>
    <span class="n">y2</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span>
    <span class="n">x3</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">4</span><span class="p">]</span>
    <span class="n">y3</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">5</span><span class="p">]</span>
    <span class="n">x4</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">6</span><span class="p">]</span>
    <span class="n">y4</span> <span class="o">=</span> <span class="n">p</span><span class="p">[</span><span class="mi">7</span><span class="p">]</span>
        
    <span class="n">t</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="n">n</span><span class="p">)</span>
    
    <span class="n">xx</span> <span class="o">=</span> <span class="n">P</span><span class="p">(</span><span class="n">t</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">p</span><span class="p">,</span> <span class="p">(</span><span class="mi">4</span><span class="p">,</span><span class="mi">2</span><span class="p">)))</span>
    
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">x1</span><span class="p">,</span> <span class="n">x4</span><span class="p">],</span> <span class="p">[</span><span class="n">y1</span><span class="p">,</span> <span class="n">y4</span><span class="p">],</span> <span class="s1">'ro'</span><span class="p">)</span> <span class="c1"># knot point</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">x1</span><span class="p">,</span> <span class="n">x2</span><span class="p">],</span> <span class="p">[</span><span class="n">y1</span><span class="p">,</span> <span class="n">y2</span><span class="p">],</span> <span class="s1">'r-'</span><span class="p">)</span> <span class="c1"># tangent</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">x3</span><span class="p">,</span> <span class="n">x4</span><span class="p">],</span> <span class="p">[</span><span class="n">y3</span><span class="p">,</span> <span class="n">y4</span><span class="p">],</span> <span class="s1">'r-'</span><span class="p">)</span> <span class="c1"># tangent</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">xx</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span> <span class="n">xx</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="s1">'-'</span><span class="p">)</span>                <span class="c1"># curve</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 5</span>
<span class="n">p</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">149</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">345</span><span class="p">,</span> <span class="mi">597</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">345</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">361</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">365</span><span class="p">,</span> <span class="mi">599</span><span class="p">,</span> <span class="mi">368</span><span class="p">,</span> <span class="mi">606</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">368</span><span class="p">,</span> <span class="mi">606</span><span class="p">,</span> <span class="mi">406</span><span class="p">,</span> <span class="mi">695</span><span class="p">,</span> <span class="mi">368</span><span class="p">,</span> <span class="mi">606</span><span class="p">,</span> <span class="mi">406</span><span class="p">,</span> <span class="mi">695</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">406</span><span class="p">,</span> <span class="mi">695</span><span class="p">,</span> <span class="mi">397</span><span class="p">,</span> <span class="mi">702</span><span class="p">,</span> <span class="mi">406</span><span class="p">,</span> <span class="mi">695</span><span class="p">,</span> <span class="mi">397</span><span class="p">,</span> <span class="mi">702</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">397</span><span class="p">,</span> <span class="mi">702</span><span class="p">,</span> <span class="mi">382</span><span class="p">,</span> <span class="mi">681</span><span class="p">,</span> <span class="mi">372</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span> <span class="mi">351</span><span class="p">,</span> <span class="mi">676</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">351</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span> <span class="mi">351</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span> <span class="mi">142</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span> <span class="mi">142</span><span class="p">,</span> <span class="mi">676</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">142</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span>  <span class="mi">33</span><span class="p">,</span> <span class="mi">439</span><span class="p">,</span> <span class="mi">142</span><span class="p">,</span> <span class="mi">676</span><span class="p">,</span>  <span class="mi">33</span><span class="p">,</span> <span class="mi">439</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">33</span><span class="p">,</span>  <span class="mi">439</span><span class="p">,</span>  <span class="mi">32</span><span class="p">,</span> <span class="mi">438</span><span class="p">,</span>  <span class="mi">32</span><span class="p">,</span> <span class="mi">436</span><span class="p">,</span>  <span class="mi">32</span><span class="p">,</span> <span class="mi">434</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">32</span><span class="p">,</span>  <span class="mi">434</span><span class="p">,</span>  <span class="mi">32</span><span class="p">,</span> <span class="mi">428</span><span class="p">,</span>  <span class="mi">35</span><span class="p">,</span> <span class="mi">426</span><span class="p">,</span>  <span class="mi">44</span><span class="p">,</span> <span class="mi">426</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">44</span><span class="p">,</span>  <span class="mi">426</span><span class="p">,</span>  <span class="mi">74</span><span class="p">,</span> <span class="mi">426</span><span class="p">,</span> <span class="mi">109</span><span class="p">,</span> <span class="mi">420</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span> <span class="mi">408</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">149</span><span class="p">,</span> <span class="mi">408</span><span class="p">,</span> <span class="mi">269</span><span class="p">,</span> <span class="mi">372</span><span class="p">,</span> <span class="mi">324</span><span class="p">,</span> <span class="mi">310</span><span class="p">,</span> <span class="mi">324</span><span class="p">,</span> <span class="mi">208</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">324</span><span class="p">,</span> <span class="mi">208</span><span class="p">,</span> <span class="mi">324</span><span class="p">,</span> <span class="mi">112</span><span class="p">,</span> <span class="mi">264</span><span class="p">,</span>  <span class="mi">37</span><span class="p">,</span> <span class="mi">185</span><span class="p">,</span>  <span class="mi">37</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">185</span><span class="p">,</span>  <span class="mi">37</span><span class="p">,</span> <span class="mi">165</span><span class="p">,</span>  <span class="mi">37</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span>  <span class="mi">44</span><span class="p">,</span> <span class="mi">119</span><span class="p">,</span>  <span class="mi">66</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">119</span><span class="p">,</span>  <span class="mi">66</span><span class="p">,</span>  <span class="mi">86</span><span class="p">,</span>  <span class="mi">90</span><span class="p">,</span>  <span class="mi">65</span><span class="p">,</span>  <span class="mi">99</span><span class="p">,</span>  <span class="mi">42</span><span class="p">,</span>  <span class="mi">99</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">42</span><span class="p">,</span>   <span class="mi">99</span><span class="p">,</span>  <span class="mi">14</span><span class="p">,</span>  <span class="mi">99</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>  <span class="mi">87</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>  <span class="mi">62</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">0</span><span class="p">,</span>    <span class="mi">62</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>  <span class="mi">24</span><span class="p">,</span>  <span class="mi">46</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">121</span><span class="p">,</span>   <span class="mi">0</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">121</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">205</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">282</span><span class="p">,</span>  <span class="mi">27</span><span class="p">,</span> <span class="mi">333</span><span class="p">,</span>  <span class="mi">78</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">333</span><span class="p">,</span>  <span class="mi">78</span><span class="p">,</span> <span class="mi">378</span><span class="p">,</span> <span class="mi">123</span><span class="p">,</span> <span class="mi">399</span><span class="p">,</span> <span class="mi">180</span><span class="p">,</span> <span class="mi">399</span><span class="p">,</span> <span class="mi">256</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">399</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">399</span><span class="p">,</span> <span class="mi">327</span><span class="p">,</span> <span class="mi">381</span><span class="p">,</span> <span class="mi">372</span><span class="p">,</span> <span class="mi">333</span><span class="p">,</span> <span class="mi">422</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">333</span><span class="p">,</span> <span class="mi">422</span><span class="p">,</span> <span class="mi">288</span><span class="p">,</span> <span class="mi">468</span><span class="p">,</span> <span class="mi">232</span><span class="p">,</span> <span class="mi">491</span><span class="p">,</span> <span class="mi">112</span><span class="p">,</span> <span class="mi">512</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">112</span><span class="p">,</span> <span class="mi">512</span><span class="p">,</span> <span class="mi">112</span><span class="p">,</span> <span class="mi">512</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span> <span class="mi">597</span><span class="p">,</span> <span class="mi">149</span><span class="p">,</span> <span class="mi">597</span><span class="p">]]</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">[</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">clf</span><span class="p">()</span>
<span class="k">for</span> <span class="n">segment</span> <span class="ow">in</span> <span class="n">p</span><span class="p">:</span>
    <span class="n">DrawBezier</span><span class="p">(</span><span class="n">segment</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">'equal'</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># T,</span>
<span class="n">p</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">237</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span> <span class="mi">120</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span> <span class="mi">120</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">237</span><span class="p">,</span> <span class="mi">120</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span>  <span class="mi">35</span><span class="p">,</span> <span class="mi">226</span><span class="p">,</span>  <span class="mi">24</span><span class="p">,</span> <span class="mi">143</span><span class="p">,</span>  <span class="mi">19</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">143</span><span class="p">,</span>  <span class="mi">19</span><span class="p">,</span> <span class="mi">143</span><span class="p">,</span>  <span class="mi">19</span><span class="p">,</span> <span class="mi">143</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">143</span><span class="p">,</span>   <span class="mi">0</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">143</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">143</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">435</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">435</span><span class="p">,</span>   <span class="mi">0</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">435</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">435</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">435</span><span class="p">,</span>  <span class="mi">19</span><span class="p">,</span> <span class="mi">435</span><span class="p">,</span>  <span class="mi">19</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">435</span><span class="p">,</span>  <span class="mi">19</span><span class="p">,</span> <span class="mi">353</span><span class="p">,</span>  <span class="mi">23</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span>  <span class="mi">36</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span> <span class="mi">109</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">339</span><span class="p">,</span> <span class="mi">109</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span> <span class="mi">108</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span> <span class="mi">620</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">339</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">339</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">393</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">393</span><span class="p">,</span> <span class="mi">620</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">393</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">507</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">529</span><span class="p">,</span> <span class="mi">602</span><span class="p">,</span> <span class="mi">552</span><span class="p">,</span> <span class="mi">492</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">552</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span> <span class="mi">552</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span> <span class="mi">576</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span> <span class="mi">576</span><span class="p">,</span> <span class="mi">492</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">576</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span> <span class="mi">576</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span> <span class="mi">570</span><span class="p">,</span> <span class="mi">662</span><span class="p">,</span> <span class="mi">570</span><span class="p">,</span> <span class="mi">662</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">570</span><span class="p">,</span> <span class="mi">662</span><span class="p">,</span> <span class="mi">570</span><span class="p">,</span> <span class="mi">662</span><span class="p">,</span>   <span class="mi">6</span><span class="p">,</span> <span class="mi">662</span><span class="p">,</span>   <span class="mi">6</span><span class="p">,</span> <span class="mi">662</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">6</span><span class="p">,</span>   <span class="mi">662</span><span class="p">,</span>   <span class="mi">6</span><span class="p">,</span> <span class="mi">662</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">492</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">0</span><span class="p">,</span>   <span class="mi">492</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span>  <span class="mi">24</span><span class="p">,</span> <span class="mi">492</span><span class="p">,</span>  <span class="mi">24</span><span class="p">,</span> <span class="mi">492</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">24</span><span class="p">,</span>  <span class="mi">492</span><span class="p">,</span>  <span class="mi">48</span><span class="p">,</span> <span class="mi">602</span><span class="p">,</span>  <span class="mi">71</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">183</span><span class="p">,</span> <span class="mi">620</span><span class="p">],</span>
     <span class="p">[</span><span class="mi">183</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">183</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span> <span class="mi">620</span><span class="p">,</span> <span class="mi">237</span><span class="p">,</span> <span class="mi">620</span><span class="p">]]</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">[</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">clf</span><span class="p">()</span>
<span class="k">for</span> <span class="n">segment</span> <span class="ow">in</span> <span class="n">p</span><span class="p">:</span>
    <span class="n">DrawBezier</span><span class="p">(</span><span class="n">segment</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">'equal'</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># f</span>
<span class="n">p</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">289</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">452</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">166</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">568</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">568</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">166</span><span class="p">,</span> <span class="mi">568</span><span class="p">,</span> <span class="mi">166</span><span class="p">,</span> <span class="mi">627</span><span class="p">,</span> <span class="mi">185</span><span class="p">,</span> <span class="mi">657</span><span class="p">,</span> <span class="mi">223</span><span class="p">,</span> <span class="mi">657</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">223</span><span class="p">,</span> <span class="mi">657</span><span class="p">,</span> <span class="mi">245</span><span class="p">,</span> <span class="mi">657</span><span class="p">,</span> <span class="mi">258</span><span class="p">,</span> <span class="mi">647</span><span class="p">,</span> <span class="mi">276</span><span class="p">,</span> <span class="mi">618</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">276</span><span class="p">,</span> <span class="mi">618</span><span class="p">,</span> <span class="mi">292</span><span class="p">,</span> <span class="mi">589</span><span class="p">,</span> <span class="mi">304</span><span class="p">,</span> <span class="mi">580</span><span class="p">,</span> <span class="mi">321</span><span class="p">,</span> <span class="mi">580</span><span class="p">],</span>
   <span class="c1">#[321, 580, 345, 580, 363, 598, 363, 621],</span>
    <span class="p">[</span><span class="mi">321</span><span class="p">,</span> <span class="mi">580</span><span class="p">,</span> <span class="mi">345</span><span class="p">,</span> <span class="mi">580</span><span class="p">,</span> <span class="mi">363</span><span class="p">,</span> <span class="mi">598</span><span class="p">,</span> <span class="mi">363</span><span class="p">,</span> <span class="mi">621</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">363</span><span class="p">,</span> <span class="mi">621</span><span class="p">,</span> <span class="mi">363</span><span class="p">,</span> <span class="mi">657</span><span class="p">,</span> <span class="mi">319</span><span class="p">,</span> <span class="mi">683</span><span class="p">,</span> <span class="mi">259</span><span class="p">,</span> <span class="mi">683</span><span class="p">],</span>
   <span class="c1">#[259, 683, 196, 683, 144, 656, 118, 611],</span>
   <span class="c1">#[118, 611,  92, 566,  84, 530,  83, 450],</span>
    <span class="p">[</span><span class="mi">259</span><span class="p">,</span> <span class="mi">683</span><span class="p">,</span> <span class="mi">196</span><span class="p">,</span> <span class="mi">683</span><span class="p">,</span> <span class="mi">126</span><span class="p">,</span> <span class="mi">666</span><span class="p">,</span> <span class="mi">100</span><span class="p">,</span> <span class="mi">621</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">621</span><span class="p">,</span>  <span class="mi">74</span><span class="p">,</span> <span class="mi">576</span><span class="p">,</span>  <span class="mi">84</span><span class="p">,</span> <span class="mi">530</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">450</span><span class="p">],</span>
    <span class="p">[</span> <span class="mi">83</span><span class="p">,</span> <span class="mi">450</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">450</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">450</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">450</span><span class="p">],</span>
    <span class="p">[</span>  <span class="mi">1</span><span class="p">,</span> <span class="mi">450</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">450</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">418</span><span class="p">],</span>
    <span class="p">[</span>  <span class="mi">1</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>   <span class="mi">1</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">418</span><span class="p">],</span>
    <span class="p">[</span> <span class="mi">83</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">104</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span> <span class="mi">104</span><span class="p">],</span>
    <span class="p">[</span> <span class="mi">83</span><span class="p">,</span> <span class="mi">104</span><span class="p">,</span>  <span class="mi">83</span><span class="p">,</span>  <span class="mi">31</span><span class="p">,</span>  <span class="mi">72</span><span class="p">,</span>  <span class="mi">19</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>  <span class="mi">15</span><span class="p">],</span>
    <span class="p">[</span>  <span class="mi">0</span><span class="p">,</span>  <span class="mi">15</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>  <span class="mi">15</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">],</span>
    <span class="p">[</span>  <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">260</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">260</span><span class="p">,</span>   <span class="mi">0</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">260</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">260</span><span class="p">,</span>   <span class="mi">0</span><span class="p">,</span> <span class="mi">260</span><span class="p">,</span>  <span class="mi">15</span><span class="p">,</span> <span class="mi">260</span><span class="p">,</span>  <span class="mi">15</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">260</span><span class="p">,</span>  <span class="mi">15</span><span class="p">,</span> <span class="mi">178</span><span class="p">,</span>  <span class="mi">18</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span>  <span class="mi">29</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span> <span class="mi">104</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">167</span><span class="p">,</span> <span class="mi">104</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span> <span class="mi">104</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span> <span class="mi">418</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">167</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">167</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">418</span><span class="p">],</span>
    <span class="p">[</span><span class="mi">289</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">418</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">452</span><span class="p">,</span> <span class="mi">289</span><span class="p">,</span> <span class="mi">452</span><span class="p">]]</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">[</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">clf</span><span class="p">()</span>
<span class="k">for</span> <span class="n">segment</span> <span class="ow">in</span> <span class="n">p</span><span class="p">:</span>
    <span class="n">DrawBezier</span><span class="p">(</span><span class="n">segment</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
<span class="n">idx</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">0</span><span class="p">],</span> <span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="o">-</span><span class="mi">2</span><span class="p">]],</span> <span class="p">[</span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">1</span><span class="p">],</span> <span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]],</span> <span class="s1">'ko'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">'equal'</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Let's-try-warping-a-segment!">Let's try warping a segment!<a class="anchor-link" href="#Let's-try-warping-a-segment!">¶</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[37]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>[321, 580, 345, 580, 363, 598, 363, 621]</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span> <span class="o">-=</span> <span class="mi">60</span>
<span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">3</span><span class="p">]</span> <span class="o">-=</span> <span class="mi">50</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">4</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">[</span><span class="mi">8</span><span class="p">,</span><span class="mi">8</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">clf</span><span class="p">()</span>
<span class="k">for</span> <span class="n">segment</span> <span class="ow">in</span> <span class="n">p</span><span class="p">:</span>
    <span class="n">DrawBezier</span><span class="p">(</span><span class="n">segment</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
<span class="n">idx</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">0</span><span class="p">],</span> <span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="o">-</span><span class="mi">2</span><span class="p">]],</span> <span class="p">[</span><span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="mi">1</span><span class="p">],</span> <span class="n">p</span><span class="p">[</span><span class="n">idx</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]],</span> <span class="s1">'ko'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">'equal'</span><span class="p">);</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3"><pre><span></span> 
</pre></div>
</div>
</div>
</div>
</div>
</div>
</div>
</body>
</html>