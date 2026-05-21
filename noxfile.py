import nox


@nox.session
def tests(session):
    """Run the pytest suite (excludes `slow` markers by default)."""
    session.install("-e", ".[test]")
    session.run("pytest", "-m", "not slow", *session.posargs)


@nox.session
def tests_all(session):
    """Run the full pytest suite including slow IEA-22 integration tests."""
    session.install("-e", ".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def cov(session):
    """Run tests with coverage report on the pynumad package."""
    session.install("-e", ".[test]")
    session.run(
        "pytest",
        "-m", "not slow",
        "--cov=pynumad",
        "--cov-report=term-missing",
        "--cov-report=html",
        *session.posargs,
    )


@nox.session
def lint(session):
    """Run ruff lint checks (style + import order + formatting)."""
    session.install("ruff")
    session.run("ruff", "check", "src")
    session.run("ruff", "format", "--check", "src")


@nox.session
def format(session):
    """Apply ruff formatting to the source tree (replaces black)."""
    session.install("ruff")
    session.run("ruff", "format", "src")
    session.run("ruff", "check", "--fix", "src")


@nox.session
def typecheck(session):
    """Run mypy (non-strict) on the package."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src/pynumad", "--ignore-missing-imports")


@nox.session
def docs(session):
    """Generate documentation."""
    session.run("pip", "install", "-e", ".")
    session.install("sphinx")
    session.install("pydata-sphinx-theme")
    session.install("sphinxcontrib-bibtex")
    session.cd("docs/")
    session.run("make", "html")


@nox.session
def serve(session):
    """Serve documentation. Port can be specified as a positional argument."""
    try:
        port = session.posargs[0]
    except IndexError:
        port = "8085"
    session.run("python", "-m", "http.server", "-b", "localhost", "-d", "docs/_build/html", port)
