<!DOCTYPE html>

<html data-content_root="../" lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/><meta content="width=device-width, initial-scale=1" name="viewport"/>
<title>Plotting the exponential function — Sphinx-Gallery 0.19.0-git documentation</title>
<script data-cfasync="false">
    document.documentElement.dataset.mode = localStorage.getItem("mode") || "";
    document.documentElement.dataset.theme = localStorage.getItem("theme") || "";
  </script>
<!--
    this give us a css class that will be invisible only if js is disabled
  -->
<noscript>
<style>
      .pst-js-only { display: none !important; }

    </style>
</noscript>
<!-- Loaded before other Sphinx assets -->
<link href="../_static/styles/theme.css?digest=8878045cc6db502f8baf" rel="stylesheet"/>
<link href="../_static/styles/pydata-sphinx-theme.css?digest=8878045cc6db502f8baf" rel="stylesheet"/>
<link href="../_static/pygments.css?v=03e43079" rel="stylesheet" type="text/css"/>
<link href="../_static/graphviz.css?v=4ae1632d" rel="stylesheet" type="text/css"/>
<link href="https://fonts.googleapis.com/css?family=Vibur" rel="stylesheet" type="text/css"/>
<link href="../_static/jupyterlite_sphinx.css?v=2c9f8f05" rel="stylesheet" type="text/css"/>
<link href="../_static/theme_override.css?v=6ecc848e" rel="stylesheet" type="text/css"/>
<link href="../_static/hide_links.css?v=60d22a59" rel="stylesheet" type="text/css"/>
<link href="../_static/sg_gallery.css?v=d2d258e8" rel="stylesheet" type="text/css"/>
<link href="../_static/sg_gallery-binder.css?v=f4aeca0c" rel="stylesheet" type="text/css"/>
<link href="../_static/sg_gallery-dataframe.css?v=2082cf3c" rel="stylesheet" type="text/css"/>
<link href="../_static/sg_gallery-rendered-html.css?v=1277b6f3" rel="stylesheet" type="text/css"/>
<!-- So that users can add custom icons -->
<script src="../_static/scripts/fontawesome.js?digest=8878045cc6db502f8baf"></script>
<!-- Pre-loaded scripts that we'll load fully later -->
<link as="script" href="../_static/scripts/bootstrap.js?digest=8878045cc6db502f8baf" rel="preload"/>
<link as="script" href="../_static/scripts/pydata-sphinx-theme.js?digest=8878045cc6db502f8baf" rel="preload"/>
<script src="../_static/documentation_options.js?v=2437ff4f"></script>
<script src="../_static/doctools.js?v=9bcbadda"></script>
<script src="../_static/sphinx_highlight.js?v=dc90522c"></script>
<script src="../_static/jupyterlite_sphinx.js?v=96e329c5"></script>
<script>DOCUMENTATION_OPTIONS.pagename = 'auto_examples/plot_1_exp';</script>
<script>
        DOCUMENTATION_OPTIONS.theme_version = '0.16.1';
        DOCUMENTATION_OPTIONS.theme_switcher_json_url = 'https://sphinx-gallery.github.io/dev/_static/switcher.json';
        DOCUMENTATION_OPTIONS.theme_switcher_version_match = 'stable';
        DOCUMENTATION_OPTIONS.show_version_warning_banner =
            false;
        </script>
<link href="../genindex.html" rel="index" title="Index"/>
<link href="../search.html" rel="search" title="Search"/>
<link href="plot_2_seaborn.html" rel="next" title="Seaborn example"/>
<link href="plot_0_sin.html" rel="prev" title="Introductory example - Plotting sin"/>
<meta content="width=device-width, initial-scale=1" name="viewport"/>
<meta content="en" name="docsearch:language"/>
<meta content="0.19.0" name="docsearch:version"/>
</head>
<body data-bs-root-margin="0px 0px -60%" data-bs-spy="scroll" data-bs-target=".bd-toc-nav" data-default-mode="" data-offset="180">
<div class="skip-link d-print-none" id="pst-skip-link"><a href="#main-content">Skip to main content</a></div>
<div id="pst-scroll-pixel-helper"></div>
<button class="btn rounded-pill" id="pst-back-to-top" type="button">
<i class="fa-solid fa-arrow-up"></i>Back to top</button>
<dialog id="pst-search-dialog">
<form action="../search.html" class="bd-search d-flex align-items-center" method="get">
<i class="fa-solid fa-magnifying-glass"></i>
<input aria-label="Search the docs ..." autocapitalize="off" autocomplete="off" autocorrect="off" class="form-control" name="q" placeholder="Search the docs ..." spellcheck="false" type="search"/>
<span class="search-button__kbd-shortcut"><kbd class="kbd-shortcut__modifier">Ctrl</kbd>+<kbd>K</kbd></span>
</form>
</dialog>
<div class="pst-async-banner-revealer d-none">
<aside aria-label="Version warning" class="d-none d-print-none" id="bd-header-version-warning"></aside>
</div>
<header class="bd-header navbar navbar-expand-lg bd-navbar d-print-none">
<div class="bd-header__inner bd-page-width">
<button aria-label="Site navigation" class="pst-navbar-icon sidebar-toggle primary-toggle">
<span class="fa-solid fa-bars"></span>
</button>
<div class="col-lg-3 navbar-header-items__start">
<div class="navbar-item">
<a class="navbar-brand logo" href="../index.html">
<p class="title logo__title">🖼️ Sphinx-Gallery</p>
</a></div>
</div>
<div class="col-lg-9 navbar-header-items">
<div class="me-auto navbar-header-items__center">
<div class="navbar-item">
<nav>
<ul class="bd-navbar-elements navbar-nav">
<li class="nav-item">
<a class="nav-link nav-internal" href="../usage.html">
    User guide
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../advanced_index.html">
    Advanced
  </a>
</li>
<li class="nav-item current active">
<a class="nav-link nav-internal" href="../galleries.html">
    Demo galleries
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../contribute.html">
    Contribution Guide
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../changes.html">
    Changelog
  </a>
</li>
</ul>
</nav></div>
</div>
<div class="navbar-header-items__end">
<div class="navbar-item navbar-persistent--container">
<button aria-label="Search" class="btn search-button-field search-button__button pst-js-only" data-bs-placement="bottom" data-bs-toggle="tooltip" title="Search">
<i class="fa-solid fa-magnifying-glass"></i>
<span class="search-button__default-text">Search</span>
<span class="search-button__kbd-shortcut"><kbd class="kbd-shortcut__modifier">Ctrl</kbd>+<kbd class="kbd-shortcut__modifier">K</kbd></span>
</button>
</div>
<div class="navbar-item">
<button aria-label="Color mode" class="btn btn-sm nav-link pst-navbar-icon theme-switch-button pst-js-only" data-bs-placement="bottom" data-bs-title="Color mode" data-bs-toggle="tooltip">
<i class="theme-switch fa-solid fa-sun fa-lg" data-mode="light" title="Light"></i>
<i class="theme-switch fa-solid fa-moon fa-lg" data-mode="dark" title="Dark"></i>
<i class="theme-switch fa-solid fa-circle-half-stroke fa-lg" data-mode="auto" title="System Settings"></i>
</button></div>
<div class="navbar-item">
<div class="version-switcher__container dropdown pst-js-only">
<button aria-controls="pst-version-switcher-list-2" aria-haspopup="listbox" aria-label="Version switcher list" class="version-switcher__button btn btn-sm dropdown-toggle" data-bs-toggle="dropdown" id="pst-version-switcher-button-2" type="button">
    Choose version  <!-- this text may get changed later by javascript -->
<span class="caret"></span>
</button>
<div aria-labelledby="pst-version-switcher-button-2" class="version-switcher__menu dropdown-menu list-group-flush py-0" id="pst-version-switcher-list-2" role="listbox">
<!-- dropdown will be populated by javascript on page load -->
</div>
</div></div>
<div class="navbar-item"><ul aria-label="Icon Links" class="navbar-icon-links">
<li class="nav-item">
<a class="nav-link pst-navbar-icon" data-bs-placement="bottom" data-bs-toggle="tooltip" href="https://github.com/sphinx-gallery/sphinx-gallery" rel="noopener" target="_blank" title="GitHub"><i aria-hidden="true" class="fa-brands fa-square-github fa-lg"></i>
<span class="sr-only">GitHub</span></a>
</li>
<li class="nav-item">
<a class="nav-link pst-navbar-icon" data-bs-placement="bottom" data-bs-toggle="tooltip" href="https://pypi.org/project/sphinx-gallery" rel="noopener" target="_blank" title="PyPI"><i aria-hidden="true" class="fa-solid fa-box fa-lg"></i>
<span class="sr-only">PyPI</span></a>
</li>
</ul></div>
</div>
</div>
<div class="navbar-persistent--mobile">
<button aria-label="Search" class="btn search-button-field search-button__button pst-js-only" data-bs-placement="bottom" data-bs-toggle="tooltip" title="Search">
<i class="fa-solid fa-magnifying-glass"></i>
<span class="search-button__default-text">Search</span>
<span class="search-button__kbd-shortcut"><kbd class="kbd-shortcut__modifier">Ctrl</kbd>+<kbd class="kbd-shortcut__modifier">K</kbd></span>
</button>
</div>
<button aria-label="On this page" class="pst-navbar-icon sidebar-toggle secondary-toggle">
<span class="fa-solid fa-outdent"></span>
</button>
</div>
</header>
<div class="bd-container">
<div class="bd-container__inner bd-page-width">
<dialog id="pst-primary-sidebar-modal"></dialog>
<div class="bd-sidebar-primary bd-sidebar" id="pst-primary-sidebar">
<div class="sidebar-header-items sidebar-primary__section">
<div class="sidebar-header-items__center">
<div class="navbar-item">
<nav>
<ul class="bd-navbar-elements navbar-nav">
<li class="nav-item">
<a class="nav-link nav-internal" href="../usage.html">
    User guide
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../advanced_index.html">
    Advanced
  </a>
</li>
<li class="nav-item current active">
<a class="nav-link nav-internal" href="../galleries.html">
    Demo galleries
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../contribute.html">
    Contribution Guide
  </a>
</li>
<li class="nav-item">
<a class="nav-link nav-internal" href="../changes.html">
    Changelog
  </a>
</li>
</ul>
</nav></div>
</div>
<div class="sidebar-header-items__end">
<div class="navbar-item">
<button aria-label="Color mode" class="btn btn-sm nav-link pst-navbar-icon theme-switch-button pst-js-only" data-bs-placement="bottom" data-bs-title="Color mode" data-bs-toggle="tooltip">
<i class="theme-switch fa-solid fa-sun fa-lg" data-mode="light" title="Light"></i>
<i class="theme-switch fa-solid fa-moon fa-lg" data-mode="dark" title="Dark"></i>
<i class="theme-switch fa-solid fa-circle-half-stroke fa-lg" data-mode="auto" title="System Settings"></i>
</button></div>
<div class="navbar-item">
<div class="version-switcher__container dropdown pst-js-only">
<button aria-controls="pst-version-switcher-list-3" aria-haspopup="listbox" aria-label="Version switcher list" class="version-switcher__button btn btn-sm dropdown-toggle" data-bs-toggle="dropdown" id="pst-version-switcher-button-3" type="button">
    Choose version  <!-- this text may get changed later by javascript -->
<span class="caret"></span>
</button>
<div aria-labelledby="pst-version-switcher-button-3" class="version-switcher__menu dropdown-menu list-group-flush py-0" id="pst-version-switcher-list-3" role="listbox">
<!-- dropdown will be populated by javascript on page load -->
</div>
</div></div>
<div class="navbar-item"><ul aria-label="Icon Links" class="navbar-icon-links">
<li class="nav-item">
<a class="nav-link pst-navbar-icon" data-bs-placement="bottom" data-bs-toggle="tooltip" href="https://github.com/sphinx-gallery/sphinx-gallery" rel="noopener" target="_blank" title="GitHub"><i aria-hidden="true" class="fa-brands fa-square-github fa-lg"></i>
<span class="sr-only">GitHub</span></a>
</li>
<li class="nav-item">
<a class="nav-link pst-navbar-icon" data-bs-placement="bottom" data-bs-toggle="tooltip" href="https://pypi.org/project/sphinx-gallery" rel="noopener" target="_blank" title="PyPI"><i aria-hidden="true" class="fa-solid fa-box fa-lg"></i>
<span class="sr-only">PyPI</span></a>
</li>
</ul></div>
</div>
</div>
<div class="sidebar-primary-items__start sidebar-primary__section">
<div class="sidebar-primary-item">
<nav aria-label="Section Navigation" class="bd-docs-nav bd-links">
<p aria-level="1" class="bd-links__title" role="heading">Section Navigation</p>
<div class="bd-toc-item navbar-nav"><ul class="current nav bd-sidenav">
<li class="toctree-l1 current active has-children"><a class="reference internal" href="index.html">Basics Gallery with Matplotlib</a><details open="open"><summary><span class="toctree-toggle" role="presentation"><i class="fa-solid fa-chevron-down"></i></span></summary><ul class="current">
<li class="toctree-l2"><a class="reference internal" href="local_module.html">Local module</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_0_sin.html">Introductory example - Plotting sin</a></li>
<li class="toctree-l2 current active"><a class="current reference internal" href="#">Plotting the exponential function</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_2_seaborn.html">Seaborn example</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_3_capture_repr.html">Capturing output representations</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_4_choose_thumbnail.html">Choosing the thumbnail figure</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_4b_provide_thumbnail.html">Providing a figure for the thumbnail image</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_5_unicode_everywhere.html">Using Unicode everywhere 🤗</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_6_function_identifier.html">Identifying function names in a script</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_7_sys_argv.html">Using <code class="docutils literal notranslate"><span class="pre">sys.argv</span></code> in examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_8_animations.html">Matplotlib animation support</a></li>
<li class="toctree-l2"><a class="reference internal" href="plot_9_multi_image_separate.html">Force plots to be displayed on separate lines</a></li>
<li class="toctree-l2 has-children"><a class="reference internal" href="no_output/index.html">No image output examples</a><details><summary><span class="toctree-toggle" role="presentation"><i class="fa-solid fa-chevron-down"></i></span></summary><ul>
<li class="toctree-l3"><a class="reference internal" href="no_output/just_code.html">A short Python script</a></li>
<li class="toctree-l3"><a class="reference internal" href="no_output/plot_raise.html">Example that fails to execute</a></li>
<li class="toctree-l3"><a class="reference internal" href="no_output/plot_raise_thumbnail.html">Example that fails to execute (with normal thumbnail behaviour)</a></li>
<li class="toctree-l3"><a class="reference internal" href="no_output/plot_strings.html">Constrained Text output frame</a></li>
<li class="toctree-l3"><a class="reference internal" href="no_output/plot_syntaxerror.html">SyntaxError</a></li>
</ul>
</details></li>
</ul>
</details></li>
<li class="toctree-l1 has-children"><a class="reference internal" href="../tutorials/index.html">Notebook-style Narrative Gallery</a><details open="open"><summary><span class="toctree-toggle" role="presentation"><i class="fa-solid fa-chevron-down"></i></span></summary><ul>
<li class="toctree-l2"><a class="reference internal" href="../tutorials/plot_parse.html">Alternating text and code</a></li>
</ul>
</details></li>
<li class="toctree-l1 has-children"><a class="reference internal" href="../auto_plotly_examples/index.html">Plotly Gallery</a><details open="open"><summary><span class="toctree-toggle" role="presentation"><i class="fa-solid fa-chevron-down"></i></span></summary><ul>
<li class="toctree-l2"><a class="reference internal" href="../auto_plotly_examples/plot_0_plotly.html">Example with the plotly graphing library</a></li>
</ul>
</details></li>
<li class="toctree-l1 has-children"><a class="reference internal" href="../auto_pyvista_examples/index.html">PyVista Gallery</a><details open="open"><summary><span class="toctree-toggle" role="presentation"><i class="fa-solid fa-chevron-down"></i></span></summary><ul>
<li class="toctree-l2"><a class="reference internal" href="../auto_pyvista_examples/plot_collisions.html">Collision</a></li>
<li class="toctree-l2"><a class="reference internal" href="../auto_pyvista_examples/plot_glyphs.html">Plotting Glyphs (Vectors or PolyData)</a></li>
<li class="toctree-l2"><a class="reference internal" href="../auto_pyvista_examples/plot_lighting.html">Lighting Properties</a></li>
<li class="toctree-l2"><a class="reference internal" href="../auto_pyvista_examples/plot_ray_trace.html">Ray Tracing</a></li>
</ul>
</details></li>
</ul>
</div>
</nav></div>
</div>
<div class="sidebar-primary-items__end sidebar-primary__section">
<div class="sidebar-primary-item">
<div class="flat" data-ea-manual="true" data-ea-publisher="readthedocs" data-ea-type="readthedocs-sidebar" id="ethical-ad-placement">
</div></div>
</div>
</div>
<main class="bd-main" id="main-content" role="main">
<div class="bd-content">
<div class="bd-article-container">
<div class="bd-header-article d-print-none">
<div class="header-article-items header-article__inner">
<div class="header-article-items__start">
<div class="header-article-item">
<nav aria-label="Breadcrumb" class="d-print-none">
<ul class="bd-breadcrumbs">
<li class="breadcrumb-item breadcrumb-home">
<a aria-label="Home" class="nav-link" href="../index.html">
<i class="fa-solid fa-home"></i>
</a>
</li>
<li class="breadcrumb-item"><a class="nav-link" href="../galleries.html">Example Galleries</a></li>
<li class="breadcrumb-item"><a class="nav-link" href="index.html">Basics Gallery with Matplotlib</a></li>
<li aria-current="page" class="breadcrumb-item active"><span class="ellipsis">Plotting the exponential function</span></li>
</ul>
</nav>
</div>
</div>
</div>
</div>
<div id="searchbox"></div>
<article class="bd-article">
<div class="sphx-glr-download-link-note admonition note">
<p class="admonition-title">Note</p>
<p><a class="reference internal" href="#sphx-glr-download-auto-examples-plot-1-exp-py"><span class="std std-ref">Go to the end</span></a>
to download the full example code. or to run this example in your browser via JupyterLite or Binder</p>
</div>
<section class="sphx-glr-example-title" id="plotting-the-exponential-function">
<span id="sphx-glr-auto-examples-plot-1-exp-py"></span><h1>Plotting the exponential function<a class="headerlink" href="#plotting-the-exponential-function" title="Link to this heading">#</a></h1>
<p>This example demonstrates how to import a local module and how images are stacked when
two plots are created in one code block (see the <a class="reference internal" href="plot_9_multi_image_separate.html"><span class="doc">Force plots to be displayed on separate lines</span></a>
example for information on controlling this behaviour). The variable <code class="docutils literal notranslate"><span class="pre">N</span></code> from the
example ‘Local module’ (file <code class="docutils literal notranslate"><span class="pre">local_module.py</span></code>) is imported in the code below.
Further, note that when there is only one code block in an example, the output appears
before the code block.</p>
<ul class="sphx-glr-horizontal">
<li><img alt="Exponential function" class="sphx-glr-multi-img" src="../_images/sphx_glr_plot_1_exp_001.png" srcset="../_images/sphx_glr_plot_1_exp_001.png, ../_images/sphx_glr_plot_1_exp_001_2_00x.png 2.00x"/></li>
<li><img alt="Negative exponential function" class="sphx-glr-multi-img" src="../_images/sphx_glr_plot_1_exp_002.png" srcset="../_images/sphx_glr_plot_1_exp_002.png, ../_images/sphx_glr_plot_1_exp_002_2_00x.png 2.00x"/></li>
</ul>
<div class="highlight-Python notranslate"><div class="highlight"><pre><span></span><span class="c1"># Code source: Óscar Nájera</span>
<span class="c1"># License: BSD 3 clause</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="c1"># You can use modules local to the example being run, here we import</span>
<span class="c1"># N from local_module</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">local_module</span><span class="w"> </span><span class="kn">import</span> <a class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance" href="https://docs.python.org/3/library/functions.html#int" title="builtins.int"><span class="n">N</span></a>  <span class="c1"># = 100</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
<span class="w">    </span><span class="sd">"""Plot exponential functions."""</span>
    <span class="n">x</span> <span class="o">=</span> <a class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-function" href="https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace" title="numpy.linspace"><span class="n">np</span><span class="o">.</span><span class="n">linspace</span></a><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <a class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance" href="https://docs.python.org/3/library/functions.html#int" title="builtins.int"><span class="n">N</span></a><span class="p">)</span>
    <span class="n">y</span> <span class="o">=</span> <a class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance" href="https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html#numpy.ufunc" title="numpy.ufunc"><span class="n">np</span><span class="o">.</span><span class="n">exp</span></a><span class="p">(</span><span class="n">x</span><span class="p">)</span>

    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure" title="matplotlib.pyplot.figure"><span class="n">plt</span><span class="o">.</span><span class="n">figure</span></a><span class="p">()</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot" title="matplotlib.pyplot.plot"><span class="n">plt</span><span class="o">.</span><span class="n">plot</span></a><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html#matplotlib.pyplot.xlabel" title="matplotlib.pyplot.xlabel"><span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span></a><span class="p">(</span><span class="s2">"$x$"</span><span class="p">)</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html#matplotlib.pyplot.ylabel" title="matplotlib.pyplot.ylabel"><span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span></a><span class="p">(</span><span class="sa">r</span><span class="s2">"$\exp(x)$"</span><span class="p">)</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html#matplotlib.pyplot.title" title="matplotlib.pyplot.title"><span class="n">plt</span><span class="o">.</span><span class="n">title</span></a><span class="p">(</span><span class="s2">"Exponential function"</span><span class="p">)</span>

    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure" title="matplotlib.pyplot.figure"><span class="n">plt</span><span class="o">.</span><span class="n">figure</span></a><span class="p">()</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot" title="matplotlib.pyplot.plot"><span class="n">plt</span><span class="o">.</span><span class="n">plot</span></a><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="o">-</span><a class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance" href="https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html#numpy.ufunc" title="numpy.ufunc"><span class="n">np</span><span class="o">.</span><span class="n">exp</span></a><span class="p">(</span><span class="o">-</span><span class="n">x</span><span class="p">))</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html#matplotlib.pyplot.xlabel" title="matplotlib.pyplot.xlabel"><span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span></a><span class="p">(</span><span class="s2">"$x$"</span><span class="p">)</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html#matplotlib.pyplot.ylabel" title="matplotlib.pyplot.ylabel"><span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span></a><span class="p">(</span><span class="sa">r</span><span class="s2">"$-\exp(-x)$"</span><span class="p">)</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html#matplotlib.pyplot.title" title="matplotlib.pyplot.title"><span class="n">plt</span><span class="o">.</span><span class="n">title</span></a><span class="p">(</span><span class="s2">"Negative exponential</span><span class="se">\n</span><span class="s2">function"</span><span class="p">)</span>
    <span class="c1"># To avoid matplotlib text output</span>
    <a class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function" href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html#matplotlib.pyplot.show" title="matplotlib.pyplot.show"><span class="n">plt</span><span class="o">.</span><span class="n">show</span></a><span class="p">()</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</pre></div>
</div>
<p class="sphx-glr-timing"><strong>Total running time of the script:</strong> (0 minutes 0.935 seconds)</p>
<p><strong>Estimated memory usage:</strong>  193 MB</p>
<div class="sphx-glr-footer sphx-glr-footer-example docutils container" id="sphx-glr-download-auto-examples-plot-1-exp-py">
<div class="binder-badge docutils container">
<a class="reference external image-reference" href="https://mybinder.org/v2/gh/sphinx-gallery/sphinx-gallery.github.io/master?urlpath=lab/tree/notebooks/auto_examples/plot_1_exp.ipynb"><img alt="Launch binder" src="../_images/binder_badge_logo.svg" style="width: 150px;"/>
</a>
</div>
<div class="lite-badge docutils container">
<a class="reference external image-reference" href="../lite/lab/index.html?path=auto_examples/plot_1_exp.ipynb"><img alt="Launch JupyterLite" src="../_images/jupyterlite_badge_logo.svg" style="width: 150px;"/>
</a>
</div>
<div class="sphx-glr-download sphx-glr-download-jupyter docutils container">
<p><a class="reference download internal" download="" href="../_downloads/4171c004075ac3b6ed7fbc36a7858d2c/plot_1_exp.ipynb"><code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">Jupyter</span> <span class="pre">notebook:</span> <span class="pre">plot_1_exp.ipynb</span></code></a></p>
</div>
<div class="sphx-glr-download sphx-glr-download-python docutils container">
<p><a class="reference download internal" download="" href="../_downloads/83f8f58e18314d50c996af36cd4c7e70/plot_1_exp.py"><code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">Python</span> <span class="pre">source</span> <span class="pre">code:</span> <span class="pre">plot_1_exp.py</span></code></a></p>
</div>
<div class="sphx-glr-download sphx-glr-download-zip docutils container">
<p><a class="reference download internal" download="" href="../_downloads/d2c53b4982f2d0b50c57c935c48be56e/plot_1_exp.zip"><code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">zipped:</span> <span class="pre">plot_1_exp.zip</span></code></a></p>
</div>
</div>
<p class="sphx-glr-signature"><a class="reference external" href="https://sphinx-gallery.github.io">Gallery generated by Sphinx-Gallery</a></p>
</section>
</article>
<footer class="prev-next-footer d-print-none">
<div class="prev-next-area">
<a class="left-prev" href="plot_0_sin.html" title="previous page">
<i class="fa-solid fa-angle-left"></i>
<div class="prev-next-info">
<p class="prev-next-subtitle">previous</p>
<p class="prev-next-title">Introductory example - Plotting sin</p>
</div>
</a>
<a class="right-next" href="plot_2_seaborn.html" title="next page">
<div class="prev-next-info">
<p class="prev-next-subtitle">next</p>
<p class="prev-next-title">Seaborn example</p>
</div>
<i class="fa-solid fa-angle-right"></i>
</a>
</div>
</footer>
</div>
<dialog id="pst-secondary-sidebar-modal"></dialog>
<div class="bd-sidebar-secondary bd-toc" id="pst-secondary-sidebar"><div class="sidebar-secondary-items sidebar-secondary__inner">
<div class="sidebar-secondary-item">
<div class="sphx-glr-sidebar-component">
<div class="sphx-glr-sidebar-item sphx-glr-download-python-sidebar" title="plot_1_exp.py">
<a download="" href="../_downloads/83f8f58e18314d50c996af36cd4c7e70/plot_1_exp.py">
<i class="fa-solid fa-download"></i>
            Download source code
          </a>
</div>
<div class="sphx-glr-sidebar-item sphx-glr-download-jupyter-sidebar" title="plot_1_exp.ipynb">
<a download="" href="../_downloads/4171c004075ac3b6ed7fbc36a7858d2c/plot_1_exp.ipynb">
<i class="fa-solid fa-download"></i>
            Download Jupyter notebook
          </a>
</div>
<div class="sphx-glr-sidebar-item sphx-glr-download-zip-sidebar" title="plot_1_exp.zip">
<a download="" href="../_downloads/d2c53b4982f2d0b50c57c935c48be56e/plot_1_exp.zip">
<i class="fa-solid fa-download"></i>
            Download zipped
          </a>
</div>
</div>
</div>
<div class="sidebar-secondary-item">
<div class="sphx-glr-sidebar-component">
<div class="sphx-glr-sidebar-item lite-badge-sidebar">
<a href="../lite/lab/index.html?path=auto_examples/plot_1_exp.ipynb">
<img alt="Launch JupyterLite" src="../_images/jupyterlite_badge_logo.svg"/>
</a>
</div>
<div class="sphx-glr-sidebar-item binder-badge-sidebar">
<a href="https://mybinder.org/v2/gh/sphinx-gallery/sphinx-gallery.github.io/master?urlpath=lab/tree/notebooks/auto_examples/plot_1_exp.ipynb">
<img alt="Launch binder" src="../_images/binder_badge_logo.svg"/>
</a>
</div>
</div>
</div>
</div></div>
</div>
<footer class="bd-footer-content">
</footer>
</main>
</div>
</div>
<!-- Scripts loaded after <body> so the DOM is not blocked -->
<script defer="" src="../_static/scripts/bootstrap.js?digest=8878045cc6db502f8baf"></script>
<script defer="" src="../_static/scripts/pydata-sphinx-theme.js?digest=8878045cc6db502f8baf"></script>
<footer class="bd-footer">
<div class="bd-footer__inner bd-page-width">
<div class="footer-items__start">
<div class="footer-item">
<p class="copyright">
    
      © Copyright 2014-2025, Sphinx-gallery developers.
      <br/>
</p>
</div>
<div class="footer-item">
<p class="sphinx-version">
    Created using <a href="https://www.sphinx-doc.org/">Sphinx</a> 8.1.3.
    <br/>
</p>
</div>
</div>
<div class="footer-items__end">
<div class="footer-item">
<p class="theme-version">
<!-- # L10n: Setting the PST URL as an argument as this does not need to be localized -->
  Built with the <a href="https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html">PyData Sphinx Theme</a> 0.16.1.
</p></div>
</div>
</div>
</footer>
</body>
</html>