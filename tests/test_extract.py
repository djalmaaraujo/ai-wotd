from wotd.sources.rss import extract_article_text

PAGE = """
<html><head><title>Gemini 4 Argon is here | AI Breakfast</title></head>
<body>
<nav><a href="/">AI Breakfast</a> <a href="/authors">Authors</a> <a href="/upgrade">Upgrade</a>
<a href="/login">Login</a> <a href="/subscribe">Subscribe</a></nav>
<article>
<h1>Gemini 4 Argon is here</h1>
<p>Google DeepMind released Gemini 4 Argon on Tuesday, a frontier model that
writes up to one million output tokens in a single response.</p>
<p>The company says Gemini 4 Argon beats every earlier Gemini release on coding
benchmarks, and it ships today in the Gemini API for all paid tiers.</p>
<p>Developers who tried the preview said the long outputs changed how they plan
agent runs, because a whole repository can now come back in one call.</p>
</article>
<footer>Subscribe to AI Breakfast. Unsubscribe any time. Terms. Privacy.</footer>
</body></html>
"""


def test_extract_article_text_keeps_the_story_and_drops_site_chrome():
    _, text = extract_article_text(PAGE)

    assert "Gemini 4 Argon on Tuesday" in text
    assert "Upgrade" not in text
    assert "Unsubscribe" not in text


def test_extract_article_text_returns_nothing_for_a_page_with_no_document():
    assert extract_article_text("<!-- only a comment -->") == ("", "")
    assert extract_article_text("   ") == ("", "")
