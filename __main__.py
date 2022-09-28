from argparse import ArgumentParser

import dicom_parser
import procedure
import visualize_data

# TODO process the arguments with argparse - based on the individual arguments of each module

parser = ArgumentParser("Studies the most recent image compression formats with medical images.")

...  # TODO add arguments here

args = parser.parse_args()

# Pre-processing
dicom_parser.main(args)
# Main processing stage
procedure.main(args)
# Post-processing
visualize_data.main(args)
