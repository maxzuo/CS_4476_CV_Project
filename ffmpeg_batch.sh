for i in $@; do
   echo "$i"
   echo "${i%%.mp4}.gif"
   ffmpeg -i "$i" "${i%%.mp4}.gif"
done
