
f = open('doc.txt', encoding='utf-8')
txt = f.read()
f.close()

txt_list = txt.split(' ')

print(txt_list)

txt_after = ''
for word in txt_list:
	if word[-2:] in ('”.'):
		txt_after += word + '  '
	elif word[-1] == '.':
		if word in ('Mr.', 'Mrs.', 'Ms.', 'Prof.', 'U.S.A.'):
			txt_after += word + ' '
		else:
			txt_after += word + '  '
	elif word[-2:] in ('.”'):
		txt_after += word + '  '
	else:
		txt_after += word + ' '

print(txt_after)
#print(txt)