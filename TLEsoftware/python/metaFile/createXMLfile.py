#!/usr/bin/python
#
import xml.etree.cElementTree as ET

root = ET.Element("root")
sat = ET.SubElement(root, "sat")

ET.SubElement(sat, "field1", name="Name").text = "Delfi-C3"
ET.SubElement(sat, "field2", name="NORADID").text = "32789"
ET.SubElement(sat, "field3", name="frequency [Hz]").text = "145870000"
ET.SubElement(sat, "field4", name="samp_rate [samp/sec]").text = "250000"
ET.SubElement(sat, "field5", name="Start recording [hh:mm]").text = "09:34"
ET.SubElement(sat, "field6", name="date [ddmmyyyy]").text = "04112015"
ET.SubElement(sat, "field7", name="Number or samples").text = "225000000"
ET.SubElement(sat, "field8", name="Recording length [sec]").text = "900"
ET.SubElement(sat, "field9", name="TLE expected TCA elevation [degree]").text = "54"
ET.SubElement(sat, "field10", name="TLE start azimuth").text = "120"
ET.SubElement(sat, "field11", name="TLE end azimuth").text = "300"
ET.SubElement(sat, "field12", name="Used TLE").text = "1 39428U 13066N   15314.38815663  .00001965  00000-0  32904-3 0  9998\n2 39428  97.6988 354.4831 0124783  66.3330 295.0913 14.65782088105159"

tree = ET.ElementTree(root)
tree.write("test_metaFile.xml")
