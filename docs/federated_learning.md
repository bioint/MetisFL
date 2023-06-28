
MetisFL SSL Connectivity Design
------------
Data useful for a machine learning problem is often generated at multiple, distributed locations. In many situations, 
these data cannot be exported from their original location due to regulatory, competitiveness, or privacy reasons. 
A primary motivating example is health records, which are heavily regulated and protected, restricting the ability to 
analyze large datasets. Industrial data (e.g., accident or safety data) is also not shared for competitiveness reasons.

The Right to be forgotten!
Given recent high-profile data leak incidents, e.g., Facebook in 2018 and Equifax in 2017, more strict data regulatory 
frameworks have been enacted in many countries, such as the European Union's General Data Protection Regulation (GDPR), 
China's Personal Information Protection Law (PIPL), and the California Consumer Privacy Act (CCPA). 
The figure below shows a sample of major data privacy bills passed across the world along with those that have 
(green color) or have not (red color) put in place legislation to secure the protection of data and privacy (green color); 

the legislation data were gathered by the United Nations Conference on Trade and Development\footnote{\url{https://unctad.org/page/data-protection-and-privacy-legislation-worldwide}}. 
In total, 137 out of 194 countries worldwide have enacted or drafted data privacy laws.

These data regulatory frameworks govern data accessing, processing, and analysis procedures and can provide protection 
for personal and proprietary data from illegal access and malicious use. When we also consider the increasing amount of 
data being generated across various disciplines, new tools are required to be introduced that will allow performing 
meaningful large-scale data analysis while at the same time providing compliance with the introduced data privacy and 
security legislations. These situations bring data distribution, security, and privacy to the forefront of the machine 
learning ecosystem and impose new challenges on how data should be processed and analyzed while respecting privacy.