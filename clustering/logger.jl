import Lumberjack
dbglog(x::Any) = Lumberjack.debug(string(x))
dbglog(x::Any...) = Lumberjack.debug(string(x))
infolog(x::Any) = Lumberjack.info(string(x))
infolog(x::Any...) = Lumberjack.info(string(x))

formatnum(x::Any; prec=4) = map((x) -> round(x, prec), x)


