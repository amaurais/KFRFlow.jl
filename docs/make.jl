using Documenter
using KFRFlow

makedocs(
    sitename = "KFRFlow",
    format = Documenter.HTML(),
    modules = [KFRFlow]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
