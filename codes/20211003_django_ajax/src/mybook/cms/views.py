from django.shortcuts import render
from django.http import HttpResponse

from cms.models import Book


from django.conf import settings
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html', {})

def ajax_number(request):
    number1 = int(request.POST.get('number1'))
    number2 = int(request.POST.get('number2'))
    plus = number1 + number2
    minus = number1 - number2
    d = {
        'plus': plus,
        'minus': minus,
    }
    return JsonResponse(d)


def book_list(request):
    """書籍の一覧"""
    # return HttpResponse('書籍の一覧')
    books = Book.objects.all().order_by('id')
    return render(request,
                  'cms/index.html',     # 使用するテンプレート
                  {'books': books})         # テンプレートに渡すデータ

def book_edit(request, book_id=None):
    """書籍の編集"""
    return HttpResponse('書籍の編集')


def book_del(request, book_id):
    """書籍の削除"""
    return HttpResponse('書籍の削除')