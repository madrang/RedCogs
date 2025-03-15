import contextlib

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
        self.old_ping_command = bot.get_command("ping")
        if self.old_ping_command:
            bot.remove_command("ping")

    async def cog_unload(self) -> None:
        if not self.old_ping_command:
            return
        with contextlib.suppress(Exception):
            self.bot.remove_command("ping")
            self.bot.add_command(self.old_ping_command)

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

    @commands.command()
    async def ping(self, ctx: commands.Context):
        """Ping command with latency information"""
        try:
            latency_ms = round(self.bot.latency * 1000)
        except OverflowError:
            await ctx.send("Connection issues in the last few seconds, precise ping times are unavailable. Try again in a minute.")
            return
        await ctx.send(f":satellite: Pong! :stopwatch:`{latency_ms}ms`")

    @staticmethod
    def get_hostname(url: str) -> str:
        try:
            urlObj = urlparse(url)
            return urlObj.hostname
        except Exception as e:
            return None

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
            addresses = []
            resolvedHostnames = []
            async for value in AsyncIter(range(5), delay=1):
                process = await asyncio.create_subprocess_shell(
                    f"ping -c 1 {url}"
                    , stdout=subprocess.PIPE
                    , stderr=subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                output = stdout.decode()
                # Time
                match = re.search(r"time=([\d.]+) ms", output)
                if not match:
                    await ctx.send("Error timedout: Unable to ping the site.")
                    return
                ping_time = float(match.group(1))
                results.append(ping_time)
                # IP Address
                match = re.search(r"\(([\d.:]+)\)", output)
                if not match:
                    continue
                if not addresses or match.group(1) not in addresses:
                    addresses.append(match.group(1))
                # Hostname
                match = re.search(r"from (\S+)", output)
                if not match:
                    continue
                if not resolvedHostnames or match.group(1) not in resolvedHostnames:
                    resolvedHostnames.append(match.group(1))

            embed = discord.Embed(
                title="Ping results for " + '\n\t\t'.join(resolvedHostnames) if resolvedHostnames else url
                , description=(f"({', '.join(addresses)})\n" if addresses else "") + f"Average Ping: {sum(results) / len(results):.2f} ms\nHighest Ping: {max(results)} ms\nLowest Ping: {min(results)} ms\n\nPing Speeds: {', '.join(map(str, results))} ms"
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
