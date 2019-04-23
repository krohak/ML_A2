if [ ! -e original/genuine ] 
then


	files=`ls original | grep '\-bc.png'`

	mkdir original/genuine

	for file in $files; do mv original/$file original/genuine; done;

	#rm *.json

	mkdir original/forged

	mv original/*.png original/forged

fi
