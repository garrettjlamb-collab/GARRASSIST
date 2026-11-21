#!/bin/zsh
# Type one MaxPreps box-score row into the focused cell of Chrome's front window.
# usage: type_row.sh <rows.json> <jersey>   (he clicks the player's Pts cell first)
vals=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["rows"][sys.argv[2]]; print(",".join("\""+str(v)+"\"" for v in r))' "$1" "$2") || exit 1
osascript <<OSA 2>&1
tell application "Google Chrome"
	set index of window 1 to 1
	activate
end tell
delay 1.5
tell application "System Events"
	set frontApp to name of first application process whose frontmost is true
	set vals to {$vals}
	repeat with i from 1 to count of vals
		set v to item i of vals
		if v is not "" then keystroke v
		delay 0.2
		if i < (count of vals) then
			key code 48
			delay 0.25
		end if
	end repeat
end tell
return "typed into: " & frontApp
OSA
