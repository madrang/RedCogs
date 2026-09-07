"""The DuckDuckGo search parser against the live result markup."""

from AgentEliza.tools.web import _SearchParser, _unwrap_duckduckgo

# The result block as the html endpoint serves it today: the title anchor,
# the icon and url anchors between, and the snippet anchor with <b> marks.
RESULT = """
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Findex.discord.red%2F&amp;rut=4692a850">Red Discord Bot - Cog Index</a>
    </h2>
    <div class="result__extras">
      <div class="result__extras__url">
        <span class="result__icon">
          <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Findex.discord.red%2F&amp;rut=4692a850">
            <img class="result__icon__img" width="16" height="16" alt="" src="//external-content.duckduckgo.com/ip3/index.discord.red.ico" name="i15" />
          </a>
        </span>
        <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Findex.discord.red%2F&amp;rut=4692a850">
          index.discord.red
        </a>
      </div>
    </div>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Findex.discord.red%2F&amp;rut=4692a850">This <b>cog</b> stores <b>discord</b> User ID&#x27;s for the purposes of mentioning the user.</a>
  </div>
</div>
"""


def test_the_parser_reads_the_live_markup() -> None:
    parser = _SearchParser()
    parser.feed(RESULT)
    assert parser.results == [
        {
            "title": "Red Discord Bot - Cog Index"
            , "url": "https://index.discord.red/"
            , "snippet": "This cog stores discord User ID's for the purposes of mentioning the user."
        }
    ]


def test_the_parser_keeps_results_apart() -> None:
    parser = _SearchParser()
    parser.feed(RESULT + RESULT)
    assert len(parser.results) == 2
    assert parser.results[0] == parser.results[1]


def test_the_redirect_unwrap() -> None:
    assert _unwrap_duckduckgo(
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=abc"
    ) == "https://example.org/a"
    assert _unwrap_duckduckgo("https://example.org/direct") == "https://example.org/direct"
