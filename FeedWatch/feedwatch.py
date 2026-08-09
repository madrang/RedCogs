import asyncio
import re

import aiohttp
import discord
from redbot.core import commands, Config


class WebDataFormatter():
    @staticmethod
    async def json(response):
        if response.status == 200:
            return await response.json()


class FeedWatch(commands.Cog):
    """FeedWatch Cog - Watch feed URLs and auto-post new entries to a Discord channel."""

    __author__ = "Madrang"
    __version__ = "0.0.1"

    def __init__(self, bot):
        self.bot = bot
        self.loaded = None
        # Init config. The identifier must stay unique and stable.
        self.config = Config.get_conf(self, identifier="feedwatch")
        self.config.register_guild(
            post_channel_id = None
            , last_post_id = 0
            , watchlist = []
        )

    async def cog_load(self) -> None:
        self.loaded = True
        # Start event loop.
        loop = self.bot.loop or asyncio.get_running_loop()
        loop.create_task(self.update_loop())

    async def cog_unload(self) -> None:
        self.loaded = False

    #
    # Red methods
    #

    def format_help_for_context(self, ctx: commands.Context) -> str:
        """Show version in help."""
        pre_processed = super().format_help_for_context(ctx)
        return f"{pre_processed}\n\nCog Version: {self.__version__}"

    async def red_get_data_for_user(self, *, user_id: int) -> None:
        """Nothing stored."""
        return

    async def red_delete_data_for_user(self, *, requester: str, user_id: int) -> None:
        """Nothing to delete."""
        return

    #
    # Addon methods
    #

    @staticmethod
    def _rendered(value) -> str:
        """Plain text out of a feed field: a string or a WP-style {"rendered": ...} dict."""
        if isinstance(value, dict):
            value = value.get("rendered", "")
        if not isinstance(value, str):
            return ""
        return re.sub(r"<[^>]+>", "", value).strip()

    @staticmethod
    def _post_id(post) -> int:
        try:
            return int(post.get("id", 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def create_embed_from_post(cls, post):
        title = cls._rendered(post.get("title")) or "New post"
        url = post.get("link") or post.get("url") or None
        description = cls._rendered(post.get("excerpt") or post.get("description"))
        embed = discord.Embed(
            title=title[:256]
            , url=url
            , description=description[:2048] or None
            , color=discord.Color.blurple()
        )
        return embed

    async def get_webdata(self, url:str, formatType:str="json"):
        formatter = getattr(WebDataFormatter, formatType, None)
        if formatter is None:
            print(f"Error missing WebDataFormatter for {formatType}")
            return
        headers = {
            "User-Agent": f"RedBot v{self.__version__} (Discord bot)"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(url) as response:
                    return await formatter(response)
            except aiohttp.ClientError as e:
                print(f"Error fetching data: {e}")

    async def send_updates_posts(self, guild_id, posts):
        post_channel_id = await self.config.guild_from_id(guild_id).post_channel_id()
        if not post_channel_id:
            return
        channel = self.bot.get_channel(post_channel_id)
        if not channel:
            return
        for post in posts:
            embed = self.create_embed_from_post(post)
            try:
                await channel.send(embed=embed)
            except discord.HTTPException as e:
                print(f"Failed to send post: {e}")

    async def update_loop(self):
        await self.bot.wait_until_ready()
        while self.loaded and not self.bot.is_closed():
            for guild in self.bot.guilds:
                guild_config = self.config.guild(guild)
                watchlist = await guild_config.watchlist()
                if not watchlist:
                    continue
                last_post_id = await guild_config.last_post_id()
                all_ids = []
                new_posts = []
                for watch_url in watchlist:
                    raw_posts = await self.get_webdata(watch_url, "json")
                    if not raw_posts:
                        continue
                    for post in raw_posts:
                        if not isinstance(post, dict):
                            continue
                        post_id = self._post_id(post)
                        if not post_id:
                            continue
                        all_ids.append(post_id)
                        if post_id > last_post_id:
                            new_posts.append(post)
                if not last_post_id and all_ids:
                    # First run: seed the marker without posting the backlog.
                    await guild_config.last_post_id.set(max(all_ids))
                elif new_posts:
                    new_posts.sort(key=self._post_id)
                    await self.send_updates_posts(guild.id, new_posts)
                    await guild_config.last_post_id.set(max(self._post_id(post) for post in new_posts))
            # Delay next check by 5 minutes
            await asyncio.sleep(300)

    @commands.command(name="setchannel", description="Set the channel for automated updates.")
    @commands.has_permissions(manage_channels=True)
    async def setchannel(self, ctx, channel: discord.TextChannel):
        await self.config.guild(ctx.guild).post_channel_id.set(channel.id)
        await ctx.send(f"Updates channel set to {channel.mention}")

    @commands.command(name="addsrc")
    async def watch_source(self, ctx: commands.Context, url: str) -> None:
        guild_config = self.config.guild(ctx.guild)
        async with guild_config.watchlist() as watchlist:
            watchlist.append(url)
        await ctx.send("The new source value has been added to the watchlist!")
