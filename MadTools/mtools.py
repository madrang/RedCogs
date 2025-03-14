import discord
from redbot.core import commands
from redbot.core.utils import AsyncIter

import asyncio
import subprocess

from urllib.parse import urlparse
import re

class MadTools(commands.Cog):
    """MadTools Cog - Experimental tools for Discord RedBot."""

    __author__ = "Madrang"
    __version__ = "0.0.1"

    def __init__(self, bot):
        self.bot = bot

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

    @staticmethod
    def get_hostname(url: str) -> str:
        try:
            urlObj = urlparse(url)
            return urlObj.hostname
        except Exception as e:
            return

    @commands.command()
    @commands.mod()
    async def pingsite(self, ctx, url: str):
        url = self.get_hostname(url)
        if not url:
            await ctx.send("Invalid or missing URL. Please provide a valid URL to ping.")
            return
        await ctx.channel.typing()
        try:
            results = []
            async for value in AsyncIter(range(5), delay=1):
                process = await asyncio.create_subprocess_shell(
                    f"ping -c 1 {url}"
                    , stdout=subprocess.PIPE
                    , stderr=subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                output = stdout.decode()
                match = re.search(r"time=([\d.]+) ms", output)
                if not match:
                    await ctx.send("Error: Unable to ping the site.")
                    return
                ping_time = float(match.group(1))
                results.append(ping_time)

            if not results:
                await ctx.send("Error: Unable to ping the site.")
                return

            embed = discord.Embed(
                title=f"Ping results for {url}"
                , description=f"Average Ping: {sum(results) / len(results):.2f} ms\nHighest Ping: {max(results)} ms\nLowest Ping: {min(results)} ms\n\nPing Speeds: {', '.join(map(str, results))} ms"
                , color=discord.Color.green()
            )
            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"An error occurred: {e}")

    @pingsite.error
    async def pingsite_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("Please provide a URL to ping.")
        elif isinstance(error, commands.BadArgument):
            await ctx.send("Please provide a valid URL or IP address.")

    @commands.command()
    @commands.mod()
    async def resolvesite(self, ctx, url: str):
        url = self.get_hostname(url)
        if not url:
            await ctx.send("Invalid or missing URL. Please provide a valid URL to resolve.")
            return
        await ctx.channel.typing()
        try:
            process = await asyncio.create_subprocess_shell(
                f"dig {url} +short"
                , stdout=subprocess.PIPE
                , stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            output = stdout.decode()
            await ctx.send(output)
        except Exception as e:
            await ctx.send(f"An error occurred: {e}")

    @resolvesite.error
    async def resolvesite_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("Please provide a URL to resolve.")
        elif isinstance(error, commands.BadArgument):
            await ctx.send("Please provide a valid URL.")
