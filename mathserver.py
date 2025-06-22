from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="math")

@mcp.tool("add")
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool("subtract")
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@mcp.tool("multiply")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")